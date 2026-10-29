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

std::string jstringToString(JNIEnv *env, jstring value) {
    if (value == nullptr) return {};
    const char *chars = env->GetStringUTFChars(value, nullptr);
    if (chars == nullptr) return {};
    std::string result(chars);
    env->ReleaseStringUTFChars(value, chars);
    return result;
}

jstring makeJavaString(JNIEnv *env, const std::string &value) {
    return env->NewStringUTF(value.c_str());
}

void releaseModelLocked() {
    if (g_sampler != nullptr) { llama_sampler_free(g_sampler); g_sampler = nullptr; }
    if (g_context != nullptr) { llama_free(g_context); g_context = nullptr; }
    if (g_model != nullptr) { llama_model_free(g_model); g_model = nullptr; }
}

} // namespace

extern "C"
JNIEXPORT jstring JNICALL
Java_com_ramzes_ai_MainActivity_nativeLoadModel(
    JNIEnv *env, jobject /* thiz */, jstring modelPath, jint requestedContext
) {
    std::lock_guard<std::mutex> lock(g_mutex);
    releaseModelLocked();

    const std::string path = jstringToString(env, modelPath);
    if (path.empty()) return makeJavaString(env, "ERROR: Путь к модели пуст.");

    LOGI("Загрузка модели: %s", path.c_str());

    llama_backend_init();

    llama_model_params modelParams = llama_model_default_params();
    modelParams.n_gpu_layers = 0; // Только CPU

    g_model = llama_model_load_from_file(path.c_str(), modelParams);
    if (g_model == nullptr) {
        LOGE("Не удалось загрузить модель");
        return makeJavaString(env, "ERROR: Не удалось загрузить модель.");
    }

    // МИНИМАЛЬНЫЕ ПАРАМЕТРЫ ДЛЯ СТАБИЛЬНОСТИ НА ANDROID
    llama_context_params contextParams = llama_context_default_params();
    contextParams.n_ctx = 512;       // Маленький контекст
    contextParams.n_batch = 1;       // БЕЗ БАТЧЕЙ (главная причина зависаний!)
    contextParams.n_ubatch = 1;
    contextParams.n_threads = 1;     // 1 поток (исключает race condition)
    contextParams.n_threads_batch = 1;

    g_context = llama_init_from_model(g_model, contextParams);
    if (g_context == nullptr) {
        LOGE("Не удалось создать контекст");
        releaseModelLocked();
        return makeJavaString(env, "ERROR: Не удалось создать контекст.");
    }

    g_sampler = llama_sampler_init_greedy();
    if (g_sampler == nullptr) {
        releaseModelLocked();
        return makeJavaString(env, "ERROR: Ошибка сэмплера");
    }

    LOGI("Модель загружена. Контекст=512, Потоков=1, Batch=1");
    return makeJavaString(env, "OK");
}

extern "C"
JNIEXPORT jstring JNICALL
Java_com_ramzes_ai_MainActivity_nativeGenerate(
    JNIEnv *env, jobject /* thiz */, jstring userPrompt, jint maxTokens
) {
    std::lock_guard<std::mutex> lock(g_mutex);

    if (!g_model || !g_context || !g_sampler) {
        return makeJavaString(env, "Ошибка: модель не загружена.");
    }

    const std::string prompt = jstringToString(env, userPrompt);
    if (prompt.empty()) return makeJavaString(env, "Введите сообщение.");

    LOGI("--- СТАРТ ГЕНЕРАЦИИ ---");
    const llama_vocab *vocab = llama_model_get_vocab(g_model);
    
    // Токенизация
    int n_tokens = llama_tokenize(vocab, prompt.c_str(), prompt.length(), nullptr, 0, true, true);
    if (n_tokens < 0) n_tokens = -n_tokens;
    
    std::vector<llama_token> tokens(n_tokens);
    n_tokens = llama_tokenize(vocab, prompt.c_str(), prompt.length(), tokens.data(), tokens.size(), true, true);
    
    if (n_tokens <= 0) {
        return makeJavaString(env, "Ошибка токенизации");
    }
    LOGI("Промпт разбит на %d токенов", n_tokens);

    // Очистка памяти и сэмплера
    llama_memory_clear(llama_get_memory(g_context), true);
    llama_sampler_reset(g_sampler);

    // ПОШАГОВОЕ ДЕКОДИРОВАНИЕ ПРОМПТА (стабильно, без батчей)
    LOGI("Начинаем обработку промпта по одному токену...");
    for (int i = 0; i < n_tokens; i++) {
        llama_batch batch = llama_batch_init(1, 0, 1);
        batch.token[0] = tokens[i];
        batch.pos[0] = i;
        batch.n_seq_id[0] = 1;
        batch.seq_id[0][0] = 0;
        batch.logits[0] = (i == n_tokens - 1); // Логиты только на последнем токене

        if (llama_decode(g_context, batch) != 0) {
            LOGE("Ошибка llama_decode на токене %d", i);
            llama_batch_free(batch);
            return makeJavaString(env, "Ошибка обработки промпта.");
        }
        llama_batch_free(batch);
    }
    LOGI("Промпт успешно обработан.");

    // ГЕНЕРАЦИЯ
    std::string answer;
    int n_past = n_tokens;
    int limit = std::min((int)maxTokens, 128);

    for (int i = 0; i < limit; i++) {
        LOGI("Генерация токена %d...", i);
        
        llama_token token = llama_sampler_sample(g_sampler, g_context, -1);

        if (llama_vocab_is_eog(vocab, token)) {
            LOGI("Достигнут конец ответа (EOG)");
            break;
        }

        char buf[256];
        int len = llama_token_to_piece(vocab, token, buf, sizeof(buf), 0, true);
        if (len > 0) {
            answer.append(buf, len);
        }

        if (i + 1 >= limit) break;

        llama_batch next = llama_batch_init(1, 0, 1);
        next.token[0] = token;
        next.pos[0] = n_past;
        next.n_seq_id[0] = 1;
        next.seq_id[0][0] = 0;
        next.logits[0] = true;

        if (llama_decode(g_context, next) != 0) {
            LOGE("Ошибка llama_decode на шаге генерации %d", i);
            llama_batch_free(next);
            break;
        }
        llama_batch_free(next);
        n_past++;
    }

    LOGI("--- КОНЕЦ ГЕНЕРАЦИИ ---");
    if (answer.empty()) {
        return makeJavaString(env, "Модель ничего не ответила (пустой ответ).");
    }
    return makeJavaString(env, answer);
}

extern "C"
JNIEXPORT void JNICALL
Java_com_ramzes_ai_MainActivity_nativeUnloadModel(JNIEnv * /* env */, jobject /* thiz */) {
    std::lock_guard<std::mutex> lock(g_mutex);
    releaseModelLocked();
    LOGI("Модель выгружена.");
}
