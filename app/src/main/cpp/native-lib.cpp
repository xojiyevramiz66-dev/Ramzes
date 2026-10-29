
#include <jni.h>
#include <android/log.h>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <mutex>
#include <string>
#include <vector>

#include "llama.h"

#define LOG_TAG "RAMZES_AI"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)

namespace {

std::mutex g_mutex;

llama_model *g_model = nullptr;
llama_context *g_context = nullptr;
llama_sampler *g_sampler = nullptr;

constexpr int DEFAULT_CONTEXT = 512;
constexpr int MAX_CONTEXT = 4096;
constexpr int MAX_GENERATION_TOKENS = 128;

std::string jstringToString(JNIEnv *env, jstring value) {
    if (value == nullptr) {
        return {};
    }

    const char *chars = env->GetStringUTFChars(value, nullptr);
    if (chars == nullptr) {
        return {};
    }

    std::string result(chars);
    env->ReleaseStringUTFChars(value, chars);
    return result;
}

jstring makeJavaString(JNIEnv *env, const std::string &value) {
    return env->NewStringUTF(value.c_str());
}

void releaseModelLocked() {
    if (g_sampler != nullptr) {
        llama_sampler_free(g_sampler);
        g_sampler = nullptr;
    }

    if (g_context != nullptr) {
        llama_free(g_context);
        g_context = nullptr;
    }

    if (g_model != nullptr) {
        llama_model_free(g_model);
        g_model = nullptr;
    }
}

// Создание полностью заполненного пакета из одного токена.
int decodeOneToken(
    llama_context *ctx,
    llama_token token,
    llama_pos position,
    bool needLogits
) {
    llama_batch batch = llama_batch_init(1, 0, 1);

    if (batch.token == nullptr ||
        batch.pos == nullptr ||
        batch.n_seq_id == nullptr ||
        batch.seq_id == nullptr ||
        batch.logits == nullptr) {
        llama_batch_free(batch);
        LOGE("Не удалось выделить память для batch");
        return -100;
    }

    // llama_batch_init выделяет память, но поля не инициализирует.
    batch.n_tokens = 1;
    batch.token[0] = token;
    batch.pos[0] = position;
    batch.n_seq_id[0] = 1;
    batch.seq_id[0][0] = 0;
    batch.logits[0] = needLogits ? 1 : 0;

    const int result = llama_decode(ctx, batch);

    llama_batch_free(batch);
    return result;
}

} // namespace

extern "C"
JNIEXPORT jstring JNICALL
Java_com_ramzes_ai_MainActivity_nativeLoadModel(
    JNIEnv *env,
    jobject /* thiz */,
    jstring modelPath,
    jint requestedContext
) {
    std::lock_guard<std::mutex> lock(g_mutex);

    releaseModelLocked();

    const std::string path = jstringToString(env, modelPath);

    if (path.empty()) {
        return makeJavaString(env, "ERROR: Путь к модели пуст.");
    }

    LOGI("Загрузка модели: %s", path.c_str());

    llama_backend_init();

    llama_model_params modelParams =
        llama_model_default_params();

    modelParams.n_gpu_layers = 0;

    g_model = llama_model_load_from_file(
        path.c_str(),
        modelParams
    );

    if (g_model == nullptr) {
        LOGE("Не удалось загрузить GGUF-модель");
        return makeJavaString(
            env,
            "ERROR: Не удалось загрузить модель. Проверьте GGUF-файл и память устройства."
        );
    }

    llama_context_params contextParams =
        llama_context_default_params();

    int contextSize = requestedContext > 0
        ? requestedContext
        : DEFAULT_CONTEXT;

    contextSize = std::max(
        128,
        std::min(contextSize, MAX_CONTEXT)
    );

    contextParams.n_ctx = static_cast<uint32_t>(contextSize);
    contextParams.n_batch = static_cast<uint32_t>(
        std::min(128, contextSize)
    );
    contextParams.n_ubatch = contextParams.n_batch;
    contextParams.n_threads = 2;
    contextParams.n_threads_batch = 2;

    g_context = llama_init_from_model(
        g_model,
        contextParams
    );

    if (g_context == nullptr) {
        LOGE("Не удалось создать контекст");
        releaseModelLocked();

        return makeJavaString(
            env,
            "ERROR: Не удалось создать контекст. Возможно, недостаточно памяти."
        );
    }

    g_sampler = llama_sampler_init_greedy();

    if (g_sampler == nullptr) {
        LOGE("Не удалось создать sampler");
        releaseModelLocked();

        return makeJavaString(
            env,
            "ERROR: Не удалось создать сэмплер."
        );
    }

    LOGI(
        "Модель загружена. Контекст=%d, CPU-потоки=2",
        contextSize
    );

    return makeJavaString(env, "OK");
}

extern "C"
JNIEXPORT jstring JNICALL
Java_com_ramzes_ai_MainActivity_nativeGenerate(
    JNIEnv *env,
    jobject /* thiz */,
    jstring userPrompt,
    jint maxTokens
) {
    std::lock_guard<std::mutex> lock(g_mutex);

    if (g_model == nullptr ||
        g_context == nullptr ||
        g_sampler == nullptr) {
        return makeJavaString(
            env,
            "Ошибка: сначала загрузите GGUF-модель."
        );
    }

    const std::string prompt = jstringToString(
        env,
        userPrompt
    );

    if (prompt.empty()) {
        return makeJavaString(env, "Введите сообщение.");
    }

    const llama_vocab *vocab = llama_model_get_vocab(g_model);

    if (vocab == nullptr) {
        return makeJavaString(
            env,
            "Ошибка: словарь модели недоступен."
        );
    }

    LOGI("Начало генерации. Длина промпта: %zu", prompt.size());

    // Первый вызов определяет необходимое количество токенов.
    int32_t tokenCount = llama_tokenize(
        vocab,
        prompt.c_str(),
        static_cast<int32_t>(prompt.size()),
        nullptr,
        0,
        true,
        true
    );

    if (tokenCount < 0) {
        if (tokenCount == std::numeric_limits<int32_t>::min()) {
            return makeJavaString(env, "Ошибка токенизации.");
        }
        tokenCount = -tokenCount;
    }

    if (tokenCount <= 0) {
        return makeJavaString(
            env,
            "Ошибка: промпт не удалось преобразовать в токены."
        );
    }

    std::vector<llama_token> tokens(
        static_cast<size_t>(tokenCount)
    );

    int32_t actualCount = llama_tokenize(
        vocab,
        prompt.c_str(),
        static_cast<int32_t>(prompt.size()),
        tokens.data(),
        static_cast<int32_t>(tokens.size()),
        true,
        true
    );

    // Повторяем токенизацию, если буфер оказался мал.
    if (actualCount < 0) {
        if (actualCount == std::numeric_limits<int32_t>::min()) {
            return makeJavaString(env, "Ошибка токенизации.");
        }

        tokens.resize(static_cast<size_t>(-actualCount));

        actualCount = llama_tokenize(
            vocab,
            prompt.c_str(),
            static_cast<int32_t>(prompt.size()),
            tokens.data(),
            static_cast<int32_t>(tokens.size()),
            true,
            true
        );
    }

    if (actualCount <= 0) {
        return makeJavaString(
            env,
            "Ошибка: токенизация промпта завершилась неудачно."
        );
    }

    tokens.resize(static_cast<size_t>(actualCount));

    const int contextSize = static_cast<int>(
        llama_n_ctx(g_context)
    );

    if (actualCount >= contextSize) {
        return makeJavaString(
            env,
            "Промпт слишком длинный. Сократите историю сообщений."
        );
    }

    int limit = std::max(
        1,
        std::min(
            static_cast<int>(maxTokens),
            MAX_GENERATION_TOKENS
        )
    );

    // Оставляем место для генерации в контексте.
    limit = std::min(limit, contextSize - actualCount);

    LOGI("Токенов промпта: %d", actualCount);
    LOGI("Доступно токенов ответа: %d", limit);

    // Каждый запрос начинает новый контекст диалога.
    llama_memory_clear(
        llama_get_memory(g_context),
        true
    );

    llama_sampler_reset(g_sampler);

    // Обработка входного промпта.
    for (int32_t i = 0; i < actualCount; ++i) {
        const bool needLogits = (i == actualCount - 1);

        const int result = decodeOneToken(
            g_context,
            tokens[static_cast<size_t>(i)],
            static_cast<llama_pos>(i),
            needLogits
        );

        if (result != 0) {
            LOGE(
                "Ошибка decode промпта: token=%d, code=%d",
                i,
                result
            );

            llama_memory_clear(
                llama_get_memory(g_context),
                true
            );

            return makeJavaString(
                env,
                "Ошибка обработки промпта (decode " +
                    std::to_string(result) + ")."
            );
        }
    }

    LOGI("Промпт обработан успешно");

    std::string answer;
    int nPast = actualCount;

    // Генерация ответа.
    for (int i = 0; i < limit; ++i) {
        const llama_token token = llama_sampler_sample(
            g_sampler,
            g_context,
            -1
        );

        if (token < 0) {
            LOGE("Сэмплер вернул недопустимый токен: %d", token);
            break;
        }

        if (llama_vocab_is_eog(vocab, token)) {
            LOGI("Достигнут конец ответа");
            break;
        }

        std::vector<char> piece(256);

        int32_t length = llama_token_to_piece(
            vocab,
            token,
            piece.data(),
            static_cast<int32_t>(piece.size()),
            0,
            true
        );

        if (length < 0) {
            if (length == std::numeric_limits<int32_t>::min()) {
                LOGE("Слишком большой фрагмент токена");
                break;
            }

            piece.resize(static_cast<size_t>(-length));

            length = llama_token_to_piece(
                vocab,
                token,
                piece.data(),
                static_cast<int32_t>(piece.size()),
                0,
                true
            );
        }

        if (length > 0) {
            answer.append(
                piece.data(),
                static_cast<size_t>(length)
            );
        } else if (length < 0) {
            LOGE("Ошибка декодирования фрагмента токена");
            break;
        }

        // Последний токен уже добавлен в ответ.
        if (i + 1 >= limit) {
            break;
        }

        const int result = decodeOneToken(
            g_context,
            token,
            static_cast<llama_pos>(nPast),
            true
        );

        if (result != 0) {
            LOGE(
                "Ошибка decode генерации: step=%d, code=%d",
                i,
                result
            );
            break;
        }

        ++nPast;
    }

    LOGI("Генерация завершена. Длина ответа: %zu", answer.size());

    if (answer.empty()) {
        return makeJavaString(
            env,
            "Модель вернула пустой ответ. Проверьте совместимость GGUF-модели."
        );
    }

    return makeJavaString(env, answer);
}

extern "C"
JNIEXPORT void JNICALL
Java_com_ramzes_ai_MainActivity_nativeUnloadModel(
    JNIEnv * /* env */,
    jobject /* thiz */
) {
    std::lock_guard<std::mutex> lock(g_mutex);

    releaseModelLocked();

    LOGI("Модель выгружена");
}
