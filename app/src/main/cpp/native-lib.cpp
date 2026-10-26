
#include <jni.h>
#include <string>
#include <vector>
#include <algorithm>
#include <mutex>
#include <cstdint>

#include "llama.h"

static llama_model* g_model = nullptr;
static llama_context* g_context = nullptr;
static std::mutex g_mutex;

extern "C"
JNIEXPORT jstring JNICALL
Java_com_ramzes_ai_MainActivity_nativeLoadModel(
        JNIEnv* env,
        jobject,
        jstring jpath,
        jint contextSize) {

    std::lock_guard<std::mutex> lock(g_mutex);

    // Освобождаем предыдущую модель, если она была загружена.
    if (g_context != nullptr) {
        llama_free(g_context);
        g_context = nullptr;
    }

    if (g_model != nullptr) {
        llama_model_free(g_model);
        g_model = nullptr;
    }

    if (jpath == nullptr) {
        return env->NewStringUTF(
            "ERROR: Путь к GGUF-модели не указан."
        );
    }

    const char* path = env->GetStringUTFChars(jpath, nullptr);

    if (path == nullptr) {
        return env->NewStringUTF(
            "ERROR: Не удалось получить путь к модели."
        );
    }

    llama_model_params modelParams =
        llama_model_default_params();

    // Совместимый базовый режим: вычисления на CPU.
    modelParams.n_gpu_layers = 0;

    g_model = llama_model_load_from_file(
        path,
        modelParams
    );

    env->ReleaseStringUTFChars(jpath, path);

    if (g_model == nullptr) {
        return env->NewStringUTF(
            "ERROR: Не удалось загрузить модель. Проверьте GGUF-файл и свободную память."
        );
    }

    llama_context_params contextParams =
        llama_context_default_params();

    const int safeContextSize =
        std::max(512, std::min(4096, static_cast<int>(contextSize)));

    contextParams.n_ctx =
        static_cast<uint32_t>(safeContextSize);

    contextParams.n_threads = 4;
    contextParams.n_threads_batch = 4;

    g_context = llama_init_from_model(
        g_model,
        contextParams
    );

    if (g_context == nullptr) {
        llama_model_free(g_model);
        g_model = nullptr;

        return env->NewStringUTF(
            "ERROR: Не удалось создать контекст. Попробуйте уменьшить его размер."
        );
    }

    return env->NewStringUTF("OK");
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
        return env->NewStringUTF(
            "Сначала загрузите GGUF-модель."
        );
    }

    if (jprompt == nullptr) {
        return env->NewStringUTF(
            "Текст запроса пуст."
        );
    }

    const char* promptChars =
        env->GetStringUTFChars(jprompt, nullptr);

    if (promptChars == nullptr) {
        return env->NewStringUTF(
            "Не удалось прочитать текст запроса."
        );
    }

    std::string prompt(promptChars);
    env->ReleaseStringUTFChars(jprompt, promptChars);

    const llama_vocab* vocab =
        llama_model_get_vocab(g_model);

    // Сначала определяем требуемое количество токенов.
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
        return env->NewStringUTF(
            "Ошибка токенизации запроса."
        );
    }

    std::vector<llama_token> tokens(tokenCount);

    int result = llama_tokenize(
        vocab,
        prompt.c_str(),
        static_cast<int32_t>(prompt.size()),
        tokens.data(),
        tokenCount,
        true,
        true
    );

    if (result < 0) {
        return env->NewStringUTF(
            "Не удалось обработать запрос."
        );
    }

    // Не допускаем переполнения контекста модели.
    const uint32_t contextCapacity = llama_n_ctx(g_context);

    if (static_cast<uint32_t>(tokenCount + 1) >= contextCapacity) {
        return env->NewStringUTF(
            "Запрос слишком длинный. Уменьшите историю сообщений."
        );
    }

    llama_memory_clear(
        llama_get_memory(g_context),
        true
    );

    llama_batch batch = llama_batch_init(
        tokenCount + 1,
        0,
        1
    );

    if (batch.token == nullptr) {
        llama_batch_free(batch);
        return env->NewStringUTF(
            "Не удалось выделить память для обработки запроса."
        );
    }

    for (int i = 0; i < tokenCount; ++i) {
        batch.token[i] = tokens[i];
        batch.pos[i] = i;
        batch.n_seq_id[i] = 1;
        batch.seq_id[i][0] = 0;
        batch.logits[i] = (i == tokenCount - 1);
    }

    batch.n_tokens = tokenCount;

    if (llama_decode(g_context, batch) != 0) {
        llama_batch_free(batch);

        return env->NewStringUTF(
            "Ошибка обработки запроса моделью."
        );
    }

    std::string output;

    const int limit = std::max(
        1,
        std::min(256, static_cast<int>(maxTokens))
    );

    for (int i = 0; i < limit; ++i) {
        float* logits = llama_get_logits_ith(
            g_context,
            -1
        );

        if (logits == nullptr) {
            break;
        }

        const int vocabSize = llama_vocab_n_tokens(vocab);
        llama_token nextToken = 0;

        // Простой жадный выбор токена с наибольшим логитом.
        for (int t = 1; t < vocabSize; ++t) {
            if (logits[t] > logits[nextToken]) {
                nextToken = t;
            }
        }

        if (llama_vocab_is_eog(vocab, nextToken)) {
            break;
        }

        char piece[512];

        int pieceSize = llama_token_to_piece(
            vocab,
            nextToken,
            piece,
            sizeof(piece),
            0,
            true
        );

        if (pieceSize > 0) {
            output.append(piece, pieceSize);
        }

        // Передаём сгенерированный токен для следующего шага.
        batch.n_tokens = 1;
        batch.token[0] = nextToken;
        batch.pos[0] = tokenCount + i;
        batch.n_seq_id[0] = 1;
        batch.seq_id[0][0] = 0;
        batch.logits[0] = true;

        if (llama_decode(g_context, batch) != 0) {
            break;
        }
    }

    llama_batch_free(batch);

    return env->NewStringUTF(output.c_str());
}

extern "C"
JNIEXPORT void JNICALL
Java_com_ramzes_ai_MainActivity_nativeUnloadModel(
        JNIEnv*,
        jobject) {

    std::lock_guard<std::mutex> lock(g_mutex);

    if (g_context != nullptr) {
        llama_free(g_context);
        g_context = nullptr;
    }

    if (g_model != nullptr) {
        llama_model_free(g_model);
        g_model = nullptr;
    }
}
