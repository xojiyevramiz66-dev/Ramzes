#include <jni.h>
#include <android/log.h>

#include <algorithm>
#include <cstdint>
#include <mutex>
#include <string>
#include <vector>

#include "llama.h"

#define LOG_TAG "RAMZES_AI"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)

static llama_model* g_model = nullptr;
static llama_context* g_context = nullptr;
static std::mutex g_mutex;

static void unloadModelLocked() {
    if (g_context != nullptr) {
        llama_free(g_context);
        g_context = nullptr;
    }

    if (g_model != nullptr) {
        llama_model_free(g_model);
        g_model = nullptr;
    }
}

static jstring makeString(JNIEnv* env, const std::string& value) {
    return env->NewStringUTF(value.c_str());
}

extern "C"
JNIEXPORT jstring JNICALL
Java_com_ramzes_ai_MainActivity_nativeLoadModel(
        JNIEnv* env,
        jobject,
        jstring jpath,
        jint contextSize) {

    std::lock_guard<std::mutex> lock(g_mutex);

    unloadModelLocked();

    if (jpath == nullptr) {
        return makeString(env, "ERROR: Путь к модели не указан.");
    }

    const char* pathChars = env->GetStringUTFChars(jpath, nullptr);
    if (pathChars == nullptr) {
        return makeString(env, "ERROR: Не удалось прочитать путь.");
    }

    const std::string path(pathChars);
    env->ReleaseStringUTFChars(jpath, pathChars);

    LOGI("Начало загрузки модели: %s", path.c_str());

    llama_model_params modelParams = llama_model_default_params();
    modelParams.n_gpu_layers = 0;

    g_model = llama_model_load_from_file(path.c_str(), modelParams);

    if (g_model == nullptr) {
        LOGE("Не удалось загрузить модель");
        return makeString(
            env,
            "ERROR: Не удалось загрузить модель. Проверьте GGUF-файл и свободную память."
        );
    }

    llama_context_params contextParams = llama_context_default_params();

    const int safeContext = std::max(
        512,
        std::min(2048, static_cast<int>(contextSize))
    );

    contextParams.n_ctx = static_cast<uint32_t>(safeContext);
    contextParams.n_threads = 4;
    contextParams.n_threads_batch = 4;

    g_context = llama_init_from_model(g_model, contextParams);

    if (g_context == nullptr) {
        LOGE("Не удалось создать контекст");
        unloadModelLocked();
        return makeString(
            env,
            "ERROR: Не удалось создать контекст. Попробуйте освободить память."
        );
    }

    LOGI("Модель загружена; контекст=%d", safeContext);
    return makeString(env, "OK");
}

extern "C"
JNIEXPORT jstring JNICALL
Java_com_ramzes_ai_MainActivity_nativeGenerate(
        JNIEnv* env,
        jobject,
        jstring jprompt,
        jint maxTokens) {

    std::lock_guard<std::mutex> lock(g_mutex);

    if (g_model == nullptr || g_context == nullptr) {
        return makeString(env, "Ошибка: сначала загрузите модель.");
    }

    if (jprompt == nullptr) {
        return makeString(env, "Ошибка: запрос пуст.");
    }

    const char* promptChars = env->GetStringUTFChars(jprompt, nullptr);
    if (promptChars == nullptr) {
        return makeString(env, "Ошибка чтения запроса.");
    }

    const std::string prompt(promptChars);
    env->ReleaseStringUTFChars(jprompt, promptChars);

    if (prompt.empty()) {
        return makeString(env, "Ошибка: запрос пуст.");
    }

    const llama_vocab* vocab = llama_model_get_vocab(g_model);

    if (vocab == nullptr) {
        return makeString(env, "Ошибка: словарь модели недоступен.");
    }

    LOGI("Токенизация запроса, длина=%zu", prompt.size());

    int tokenCount = -llama_tokenize(
        vocab,
        prompt.c_str(),
        static_cast<int32_t>(prompt.size()),
        nullptr,
        0,
        true,
        true
    );

    if (tokenCount <= 0) {
        LOGE("Не удалось определить число токенов: %d", tokenCount);
        return makeString(env, "Ошибка токенизации запроса.");
    }

    const uint32_t contextCapacity = llama_n_ctx(g_context);

    if (static_cast<uint32_t>(tokenCount + 2) >= contextCapacity) {
        return makeString(
            env,
            "Запрос слишком длинный. Очистите историю чата и повторите."
        );
    }

    std::vector<llama_token> tokens(
        static_cast<size_t>(tokenCount)
    );

    const int actualCount = llama_tokenize(
        vocab,
        prompt.c_str(),
        static_cast<int32_t>(prompt.size()),
        tokens.data(),
        tokenCount,
        true,
        true
    );

    if (actualCount < 0) {
        return makeString(env, "Ошибка преобразования текста в токены.");
    }

    tokenCount = actualCount;
    tokens.resize(static_cast<size_t>(tokenCount));

    if (tokenCount == 0) {
        return makeString(env, "Запрос не содержит токенов.");
    }

    llama_memory_clear(llama_get_memory(g_context), true);

    llama_batch batch = llama_batch_init(tokenCount + 1, 0, 1);

    if (batch.token == nullptr ||
        batch.pos == nullptr ||
        batch.n_seq_id == nullptr ||
        batch.seq_id == nullptr ||
        batch.logits == nullptr) {

        llama_batch_free(batch);
        return makeString(env, "Не удалось выделить память для генерации.");
    }

    for (int i = 0; i < tokenCount; ++i) {
        batch.token[i] = tokens[i];
        batch.pos[i] = i;
        batch.n_seq_id[i] = 1;
        batch.seq_id[i][0] = 0;
        batch.logits[i] = (i == tokenCount - 1);
    }

    batch.n_tokens = tokenCount;

    LOGI("Обработка входных токенов: %d", tokenCount);

    const int promptResult = llama_decode(g_context, batch);

    if (promptResult != 0) {
        LOGE("llama_decode(prompt) вернул %d", promptResult);
        llama_batch_free(batch);
        return makeString(env, "Ошибка обработки запроса моделью.");
    }

    const int limit = std::max(
        1,
        std::min(64, static_cast<int>(maxTokens))
    );

    const int vocabSize = llama_vocab_n_tokens(vocab);
    std::string output;

    LOGI("Начало генерации, максимум токенов=%d", limit);

    for (int i = 0; i < limit; ++i) {
        const int position = tokenCount + i;

        if (position >= static_cast<int>(contextCapacity)) {
            LOGI("Достигнут предел контекста");
            break;
        }

        float* logits = llama_get_logits_ith(g_context, -1);

        if (logits == nullptr) {
            LOGE("Логиты недоступны");
            break;
        }

        llama_token nextToken = 0;

        for (int t = 1; t < vocabSize; ++t) {
            if (logits[t] > logits[nextToken]) {
                nextToken = t;
            }
        }

        if (llama_vocab_is_eog(vocab, nextToken)) {
            LOGI("Получен токен завершения");
            break;
        }

        char piece[1024];

        const int pieceSize = llama_token_to_piece(
            vocab,
            nextToken,
            piece,
            static_cast<int32_t>(sizeof(piece)),
            0,
            true
        );

        if (pieceSize > 0) {
            output.append(piece, static_cast<size_t>(pieceSize));
        }

        LOGI("Сгенерирован токен %d из %d", i + 1, limit);

        // Не обрабатываем последний токен повторно:
        // для следующей итерации он должен пройти через llama_decode.
        batch.n_tokens = 1;
        batch.token[0] = nextToken;
        batch.pos[0] = position;
        batch.n_seq_id[0] = 1;
        batch.seq_id[0][0] = 0;
        batch.logits[0] = true;

        if (i + 1 < limit) {
            const int decodeResult = llama_decode(g_context, batch);

            if (decodeResult != 0) {
                LOGE("llama_decode(token) вернул %d", decodeResult);
                break;
            }
        }
    }

    llama_batch_free(batch);

    LOGI("Генерация завершена, длина ответа=%zu", output.size());

    if (output.empty()) {
        return makeString(
            env,
            "Модель не вернула текст. Проверьте совместимость модели и шаблона чата."
        );
    }

    return makeString(env, output);
}

extern "C"
JNIEXPORT void JNICALL
Java_com_ramzes_ai_MainActivity_nativeUnloadModel(
        JNIEnv*,
        jobject) {

    std::lock_guard<std::mutex> lock(g_mutex);
    LOGI("Выгрузка модели");
    unloadModelLocked();
}
