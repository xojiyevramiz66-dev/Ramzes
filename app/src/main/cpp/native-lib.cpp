
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

static llama_model *g_model = nullptr;
static llama_context *g_context = nullptr;
static llama_sampler *g_sampler = nullptr;

static std::mutex g_mutex;

static constexpr int32_t CONTEXT_SIZE = 1024;
static constexpr int32_t BATCH_SIZE = 64;
static constexpr int32_t MAX_PROMPT_TOKENS = 768;
static constexpr int32_t MAX_GENERATION_TOKENS = 64;

static std::string jstringToString(JNIEnv *env, jstring value) {
    if (value == nullptr) return {};

    const char *chars = env->GetStringUTFChars(value, nullptr);
    if (chars == nullptr) return {};

    std::string result(chars);
    env->ReleaseStringUTFChars(value, chars);
    return result;
}

static jstring makeJavaString(JNIEnv *env, const std::string &value) {
    return env->NewStringUTF(value.c_str());
}

static void releaseModelLocked() {
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

static bool tokenizePrompt(
    const llama_vocab *vocab,
    const std::string &prompt,
    std::vector<llama_token> &tokens
) {
    const int32_t required = llama_tokenize(
        vocab,
        prompt.c_str(),
        static_cast<int32_t>(prompt.size()),
        nullptr,
        0,
        true,
        true
    );

    if (required >= 0) {
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
        static_cast<int32_t>(prompt.size()),
        tokens.data(),
        capacity,
        true,
        true
    );

    if (actual < 0) {
        LOGE("Ошибка токенизации: %d", actual);
        return false;
    }

    tokens.resize(static_cast<size_t>(actual));
    return !tokens.empty();
}

static bool decodePromptInBatches(
    llama_context *context,
    const std::vector<llama_token> &tokens
) {
    const int32_t total = static_cast<int32_t>(tokens.size());

    for (int32_t offset = 0; offset < total; offset += BATCH_SIZE) {
        const int32_t count = std::min(BATCH_SIZE, total - offset);
        llama_batch batch = llama_batch_init(count, 0, 1);

        if (batch.token == nullptr ||
            batch.pos == nullptr ||
            batch.n_seq_id == nullptr ||
            batch.seq_id == nullptr ||
            batch.logits == nullptr) {
            LOGE("Не удалось выделить память для батча");
            llama_batch_free(batch);
            return false;
        }

        batch.n_tokens = count;

        for (int32_t j = 0; j < count; ++j) {
            const int32_t index = offset + j;

            batch.token[j] = tokens[index];
            batch.pos[j] = index;
            batch.n_seq_id[j] = 1;
            batch.seq_id[j][0] = 0;
            batch.logits[j] = (index == total - 1);
        }

        LOGI("Обработка промпта: %d/%d токенов",
             std::min(offset + count, total), total);

        const int result = llama_decode(context, batch);
        llama_batch_free(batch);

        if (result != 0) {
            LOGE("llama_decode: ошибка %d на позиции %d", result, offset);
            return false;
        }
    }

    return true;
}

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
        return makeJavaString(env, "ERROR: Путь к модели пуст");
    }

    LOGI("Начало загрузки модели: %s", path.c_str());
    LOGI("Запрошенный контекст: %d", requestedContext);

    llama_backend_init();

    llama_model_params modelParams = llama_model_default_params();
    modelParams.n_gpu_layers = 0;

    g_model = llama_model_load_from_file(path.c_str(), modelParams);
    if (g_model == nullptr) {
        LOGE("Не удалось загрузить GGUF-модель");
        releaseModelLocked();
        return makeJavaString(
            env,
            "ERROR: Не удалось загрузить модель. Проверьте GGUF и Logcat."
        );
    }

    const int32_t n_ctx = std::max(
        512,
        std::min(static_cast<int32_t>(requestedContext), CONTEXT_SIZE)
    );

    llama_context_params contextParams = llama_context_default_params();
    contextParams.n_ctx = n_ctx;
    contextParams.n_batch = BATCH_SIZE;
    contextParams.n_ubatch = BATCH_SIZE;
    contextParams.n_threads = 4;
    contextParams.n_threads_batch = 4;

    g_context = llama_init_from_model(g_model, contextParams);
    if (g_context == nullptr) {
        LOGE("Не удалось создать llama_context");
        releaseModelLocked();
        return makeJavaString(
            env,
            "ERROR: Не удалось создать контекст. Проверьте память устройства."
        );
    }

    g_sampler = llama_sampler_init_greedy();
    if (g_sampler == nullptr) {
        LOGE("Не удалось создать sampler");
        releaseModelLocked();
        return makeJavaString(env, "ERROR: Не удалось создать sampler");
    }

    LOGI("Модель загружена; контекст=%d, батч=%d", n_ctx, BATCH_SIZE);
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

    if (g_model == nullptr || g_context == nullptr || g_sampler == nullptr) {
        return makeJavaString(env, "Ошибка: модель не загружена.");
    }

    const std::string prompt = jstringToString(env, userPrompt);
    if (prompt.empty()) {
        return makeJavaString(env, "Введите сообщение.");
    }

    const auto start = std::chrono::steady_clock::now();
    LOGI("Начало генерации");

    const llama_vocab *vocab = llama_model_get_vocab(g_model);
    std::vector<llama_token> tokens;

    if (!tokenizePrompt(vocab, prompt, tokens)) {
        return makeJavaString(env, "Ошибка токенизации промпта.");
    }

    if (tokens.size() > MAX_PROMPT_TOKENS) {
        LOGE("Слишком длинный промпт: %zu токенов", tokens.size());
        return makeJavaString(
            env,
            "Сообщение слишком длинное. Сократите историю или запрос."
        );
    }

    llama_memory_t memory = llama_get_memory(g_context);
    if (memory != nullptr) {
        llama_memory_clear(memory, true);
    }

    if (!decodePromptInBatches(g_context, tokens)) {
        return makeJavaString(
            env,
            "Ошибка обработки промпта. Проверьте Logcat."
        );
    }

    const int32_t limit = std::max(
        1,
        std::min(static_cast<int32_t>(maxTokens), MAX_GENERATION_TOKENS)
    );

    std::string answer;
    answer.reserve(static_cast<size_t>(limit) * 8);

    for (int32_t i = 0; i < limit; ++i) {
        const llama_token token =
            llama_sampler_sample(g_sampler, g_context, -1);

        llama_sampler_accept(g_sampler, token);

        if (llama_vocab_is_eog(vocab, token)) {
            LOGI("Конец ответа на токене %d", i);
            break;
        }

        char piece[4096];

        const int32_t pieceSize = llama_token_to_piece(
            vocab,
            token,
            piece,
            sizeof(piece),
            0,
            true
        );

        if (pieceSize < 0) {
            LOGE("Ошибка преобразования токена: %d", pieceSize);
            break;
        }

        if (pieceSize > 0) {
            answer.append(piece, static_cast<size_t>(pieceSize));
        }

        if (i + 1 >= limit) break;

        llama_batch next = llama_batch_init(1, 0, 1);

        if (next.token == nullptr ||
            next.pos == nullptr ||
            next.n_seq_id == nullptr ||
            next.seq_id == nullptr ||
            next.logits == nullptr) {
            LOGE("Не удалось выделить батч для генерации");
            llama_batch_free(next);
            break;
        }

        next.n_tokens = 1;
        next.token[0] = token;
        next.pos[0] = static_cast<llama_pos>(tokens.size() + i);
        next.n_seq_id[0] = 1;
        next.seq_id[0][0] = 0;
        next.logits[0] = true;

        const int result = llama_decode(g_context, next);
        llama_batch_free(next);

        if (result != 0) {
            LOGE("Ошибка декодирования токена %d: %d", i, result);
            break;
        }
    }

    const auto elapsed =
        std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - start
        ).count();

    LOGI("Генерация завершена: %zu байт, время=%lld мс",
         answer.size(), static_cast<long long>(elapsed));

    if (answer.empty()) {
        return makeJavaString(
            env,
            "Модель не вернула текст. Проверьте Logcat и шаблон чата."
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
