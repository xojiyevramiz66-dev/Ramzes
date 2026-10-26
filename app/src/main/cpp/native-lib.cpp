
#include <jni.h>
#include <string>
#include <vector>
#include <algorithm>
#include <mutex>

#include "llama.h"

static llama_model* g_model = nullptr;
static llama_context* g_context = nullptr;
static std::mutex g_mutex;

extern "C"
JNIEXPORT jstring JNICALL
Java_com_dumdam_ai_MainActivity_nativeLoadModel(
        JNIEnv* env,
        jobject,
        jstring jpath,
        jint contextSize) {

    std::lock_guard<std::mutex> lock(g_mutex);

    if (g_context != nullptr) {
        llama_free(g_context);
        g_context = nullptr;
    }

    if (g_model != nullptr) {
        llama_model_free(g_model);
        g_model = nullptr;
    }

    const char* path = env->GetStringUTFChars(jpath, nullptr);

    llama_model_params modelParams =
        llama_model_default_params();

    modelParams.n_gpu_layers = 0;

    g_model = llama_model_load_from_file(
        path,
        modelParams
    );

    env->ReleaseStringUTFChars(jpath, path);

    if (g_model == nullptr) {
        return env->NewStringUTF(
            "ERROR: Не удалось загрузить GGUF-модель."
        );
    }

    llama_context_params contextParams =
        llama_context_default_params();

    contextParams.n_ctx =
        static_cast<uint32_t>(
            std::max(512, std::min(4096, (int)contextSize))
        );

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
            "ERROR: Не удалось создать контекст модели."
        );
    }

    return env->NewStringUTF("OK");
}

extern "C"
JNIEXPORT jstring JNICALL
Java_com_dumdam_ai_MainActivity_nativeGenerate(
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

    const char* promptChars =
        env->GetStringUTFChars(jprompt, nullptr);

    std::string prompt(promptChars);

    env->ReleaseStringUTFChars(jprompt, promptChars);

    const llama_vocab* vocab =
        llama_model_get_vocab(g_model);

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

    llama_memory_clear(
        llama_get_memory(g_context),
        true
    );

    llama_batch batch = llama_batch_init(
        tokenCount + 1,
        0,
        1
    );

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

        int vocabSize = llama_vocab_n_tokens(vocab);
        llama_token nextToken = 0;

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
Java_com_dumdam_ai_MainActivity_nativeUnloadModel(
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
