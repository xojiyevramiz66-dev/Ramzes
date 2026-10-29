#include <jni.h>
#include <android/log.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
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

bool g_backend_initialized = false;

constexpr int32_t DEFAULT_CONTEXT_SIZE = 1024;
constexpr int32_t BATCH_SIZE = 64;
constexpr int32_t MAX_PROMPT_TOKENS = 768;
constexpr int32_t MAX_GENERATION_TOKENS = 64;
constexpr int32_t THREAD_COUNT = 4;

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

/*
 * llama_tokenize возвращает отрицательное число,
 * когда переданный буфер слишком мал.
 */
bool tokenizePrompt(
    const llama_vocab *vocab,
    const std::string &prompt,
    std::vector<llama_token> &tokens
) {
    if (vocab == nullptr || prompt.empty()) {
        LOGE("Пустой промпт или словарь модели");
        return false;
    }

    if (prompt.size() > static_cast<size_t>(INT32_MAX)) {
        LOGE("Промпт слишком велик");
        return false;
    }

    const int32_t promptLength =
        static_cast<int32_t>(prompt.size());

    int32_t required = llama_tokenize(
        vocab,
        prompt.c_str(),
        promptLength,
        nullptr,
        0,
        true,
        true
    );

    if (required >= 0) {
        // Для пустого буфера ожидается отрицательный размер.
        // Если API вернул ноль, не продолжаем с пустым контекстом.
        LOGE("Не удалось определить размер токенизации: %d", required);
        return false;
    }

    const int32_t capacity = -required;

    if (capacity <= 0 || capacity > 100000) {
        LOGE("Недопустимый размер токенизации: %d", capacity);
        return false;
    }

    tokens.resize(static_cast<size_t>(capacity));

    const int32_t actual = llama_tokenize(
        vocab,
        prompt.c_str(),
        promptLength,
        tokens.data(),
        capacity,
        true,
        true
    );

    if (actual < 0) {
        LOGE("Ошибка токенизации: %d", actual);
        tokens.clear();
        return false;
    }

    tokens.resize(static_cast<size_t>(actual));

    if (tokens.empty()) {
        LOGE("После токенизации не осталось токенов");
        return false;
    }

    return true;
}

/*
 * Обрабатываем промпт небольшими батчами, чтобы не требовать
 * большой временный буфер для всего запроса сразу.
 */
bool decodePromptInBatches(
    llama_context *context,
    const std::vector<llama_token> &tokens
) {
    if (context == nullptr || tokens.empty()) {
        return false;
    }

    const int32_t total = static_cast<int32_t>(tokens.size());

    for (int32_t offset = 0; offset < total; offset += BATCH_SIZE) {
        const int32_t count =
            std::min(BATCH_SIZE, total - offset);

        llama_batch batch = llama_batch_init(count, 0, 1);

        if (
            batch.token == nullptr ||
            batch.pos == nullptr ||
            batch.n_seq_id == nullptr ||
            batch.seq_id == nullptr ||
            batch.logits == nullptr
        ) {
            LOGE("Не удалось выделить память для батча");
            llama_batch_free(batch);
            return false;
        }

        batch.n_tokens = count;

        for (int32_t j = 0; j < count; ++j) {
            const int32_t index = offset + j;

            batch.token[j] = tokens[static_cast<size_t>(index)];
            batch.pos[j] = static_cast<llama_pos>(index);
            batch.n_seq_id[j] = 1;
            batch.seq_id[j][0] = 0;

            // Логиты нужны только для последнего токена промпта.
            batch.logits[j] = (index == total - 1);
        }

        const int result = llama_decode(context, batch);

        llama_batch_free(batch);

        if (result != 0) {
            LOGE(
                "llama_decode завершился с ошибкой %d на позиции %d",
                result,
                offset
            );
            return false;
        }
    }

    return true;
}

bool initializeBackend() {
    if (!g_backend_initialized) {
        llama_backend_init();
        g_backend_initialized = true;
    }

    return true;
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

    // Удаляем предыдущую модель перед новой загрузкой.
    releaseModelLocked();

    const std::string path = jstringToString(env, modelPath);

    if (path.empty()) {
        return makeJavaString(
            env,
            "ERROR: Путь к модели пуст."
        );
    }

    LOGI("Загрузка модели: %s", path.c_str());

    try {
        initializeBackend();

        llama_model_params modelParams =
            llama_model_default_params();

        // CPU-режим — наиболее предсказуемый вариант для Android.
        modelParams.n_gpu_layers = 0;

        g_model = llama_model_load_from_file(
            path.c_str(),
            modelParams
        );

        if (g_model == nullptr) {
            LOGE("Не удалось загрузить модель из файла");

            releaseModelLocked();

            return makeJavaString(
                env,
                "ERROR: Не удалось загрузить модель. " 
                "Проверьте файл GGUF и свободную память."
            );
        }

        int32_t contextSize = requestedContext;

        if (contextSize <= 0) {
            contextSize = DEFAULT_CONTEXT_SIZE;
        }

        contextSize = std::max(
            512,
            std::min(contextSize, DEFAULT_CONTEXT_SIZE)
        );

        llama_context_params contextParams =
            llama_context_default_params();

        contextParams.n_ctx = static_cast<uint32_t>(contextSize);
        contextParams.n_batch = BATCH_SIZE;
        contextParams.n_ubatch = BATCH_SIZE;
        contextParams.n_threads = THREAD_COUNT;
        contextParams.n_threads_batch = THREAD_COUNT;

        g_context = llama_init_from_model(
            g_model,
            contextParams
        );

        if (g_context == nullptr) {
            LOGE("Не удалось создать контекст");

            releaseModelLocked();

            return makeJavaString(
                env,
                "ERROR: Не удалось создать контекст. "
                "Попробуйте меньшую модель."
            );
        }

        g_sampler = llama_sampler_init_greedy();

        if (g_sampler == nullptr) {
            LOGE("Не удалось создать sampler");

            releaseModelLocked();

            return makeJavaString(
                env,
                "ERROR: Не удалось создать генератор токенов."
            );
        }

        LOGI(
            "Модель загружена. Контекст=%d, batch=%d",
            contextSize,
            BATCH_SIZE
        );

        return makeJavaString(env, "OK");
    } catch (...) {
        LOGE("Неизвестная ошибка во время загрузки модели");

        releaseModelLocked();

        return makeJavaString(
            env,
            "ERROR: Исключение при загрузке модели."
        );
    }
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

    if (
        g_model == nullptr ||
        g_context == nullptr ||
        g_sampler == nullptr
    ) {
        return makeJavaString(
            env,
            "Ошибка: модель не загружена."
        );
    }

    const std::string prompt = jstringToString(env, userPrompt);

    if (prompt.empty()) {
        return makeJavaString(env, "Введите сообщение.");
    }

    const auto start = std::chrono::steady_clock::now();

    LOGI("Начало генерации");

    const llama_vocab *vocab =
        llama_model_get_vocab(g_model);

    std::vector<llama_token> tokens;

    if (!tokenizePrompt(vocab, prompt, tokens)) {
        return makeJavaString(
            env,
            "Ошибка токенизации промпта."
        );
    }

    if (tokens.size() > MAX_PROMPT_TOKENS) {
        LOGE(
            "Промпт слишком длинный: %zu токенов",
            tokens.size()
        );

        return makeJavaString(
            env,
            "Запрос слишком длинный. "
            "Сократите сообщение или историю чата."
        );
    }

    const int32_t generationLimit = std::max(
        1,
        std::min(
            static_cast<int32_t>(maxTokens),
            MAX_GENERATION_TOKENS
        )
    );

    /*
     * Проверяем, что промпт и генерация помещаются в контекст.
     * Не начинаем генерацию, если позиции токенов превысят лимит.
     */
    const uint32_t contextSize = llama_n_ctx(g_context);

    if (
        tokens.size() + static_cast<size_t>(generationLimit)
        > static_cast<size_t>(contextSize)
    ) {
        return makeJavaString(
            env,
            "Недостаточно места в контексте. "
            "Сократите историю сообщений."
        );
    }

    llama_memory_t memory = llama_get_memory(g_context);

    if (memory != nullptr) {
        llama_memory_clear(memory, true);
    }

    // Сбрасываем состояние sampler перед каждым новым ответом.
    llama_sampler_reset(g_sampler);

    if (!decodePromptInBatches(g_context, tokens)) {
        if (memory != nullptr) {
            llama_memory_clear(memory, true);
        }

        return makeJavaString(
            env,
            "Ошибка обработки промпта. Проверьте версию llama.cpp."
        );
    }

    std::string answer;
    answer.reserve(static_cast<size_t>(generationLimit) * 8);

    bool decodeFailed = false;

    for (int32_t i = 0; i < generationLimit; ++i) {
        const llama_token token =
            llama_sampler_sample(g_sampler, g_context, -1);

        if (llama_vocab_is_eog(vocab, token)) {
            LOGI("Модель завершила ответ на токене %d", i);
            break;
        }

        llama_sampler_accept(g_sampler, token);

        char piece[4096];

        const int32_t pieceSize = llama_token_to_piece(
            vocab,
            token,
            piece,
            static_cast<int32_t>(sizeof(piece)),
            0,
            true
        );

        if (pieceSize < 0) {
            LOGE("Ошибка преобразования токена: %d", pieceSize);
            decodeFailed = true;
            break;
        }

        if (pieceSize > 0) {
            answer.append(
                piece,
                static_cast<size_t>(pieceSize)
            );
        }

        // Для последнего токена следующий decode не нужен.
        if (i + 1 >= generationLimit) {
            break;
        }

        llama_batch next = llama_batch_init(1, 0, 1);

        if (
            next.token == nullptr ||
            next.pos == nullptr ||
            next.n_seq_id == nullptr ||
            next.seq_id == nullptr ||
            next.logits == nullptr
        ) {
            LOGE("Не удалось выделить батч для генерации");
            llama_batch_free(next);
            decodeFailed = true;
            break;
        }

        next.n_tokens = 1;
        next.token[0] = token;
        next.pos[0] = static_cast<llama_pos>(
            tokens.size() + static_cast<size_t>(i)
        );
        next.n_seq_id[0] = 1;
        next.seq_id[0][0] = 0;
        next.logits[0] = true;

        const int result = llama_decode(g_context, next);

        llama_batch_free(next);

        if (result != 0) {
            LOGE(
                "Ошибка декодирования токена %d: %d",
                i,
                result
            );

            decodeFailed = true;
            break;
        }
    }

    const auto elapsed =
        std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - start
        ).count();

    LOGI(
        "Генерация завершена: %zu байт, %lld мс",
        answer.size(),
        static_cast<long long>(elapsed)
    );

    if (decodeFailed && answer.empty()) {
        if (memory != nullptr) {
            llama_memory_clear(memory, true);
        }

        return makeJavaString(
            env,
            "Ошибка генерации. Проверьте Logcat и версию llama.cpp."
        );
    }

    if (answer.empty()) {
        return makeJavaString(
            env,
            "Модель не вернула текст. "
            "Проверьте совместимость шаблона чата."
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

    LOGI("Модель освобождена");
}
