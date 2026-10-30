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

    // Оптимальные параметры для Android (взяты из Ponko)
    llama_context_params contextParams = llama_context_default_params();
    contextParams.n_ctx = 4096;       // Контекст как в Ponko
    contextParams.n_batch = 512;      // Батч как в Ponko
    contextParams.n_ubatch = 512;     
    contextParams.n_threads = 4;      
    contextParams.n_threads_batch = 4;

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

    LOGI("Модель загружена. Контекст=4096, Batch=512, Потоков=4");
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

    // 1. Получаем сырой текст (историю чата) от Kotlin
    const std::string rawText = jstringToString(env, userPrompt);
    if (rawText.empty()) return makeJavaString(env, "Введите сообщение.");

    LOGI("--- СТАРТ ГЕНЕРАЦИИ ---");
    
    // 2. Применяем шаблон чата, вшитый в модель (ChatML, Llama3, Gemma и т.д.)
    const char *tmpl = llama_model_chat_template(g_model, nullptr);
    std::string prompt;
    
    if (tmpl != nullptr) {
        // Используем встроенный шаблон (как это делает Ponko)
        std::vector<char> formatted(8192);
        int n = llama_chat_apply_template(tmpl, rawText.c_str(), rawText.length(), true, formatted.data(), formatted.size());
        if (n > 0) {
            prompt.assign(formatted.data(), n);
            LOGI("Применен встроенный шаблон чата.");
        } else {
            prompt = rawText; // Фолбэк
            LOGI("Не удалось применить шаблон, используем сырой текст.");
        }
    } else {
        // Если в модели нет шаблона, используем сырой текст
        prompt = rawText;
        LOGI("Шаблон чата не найден, используем сырой текст.");
    }

    const llama_vocab *vocab = llama_model_get_vocab(g_model);
    
    // 3. Токенизация
    int n_tokens = llama_tokenize(vocab, prompt.c_str(), prompt.length(), nullptr, 0, true, true);
    if (n_tokens < 0) n_tokens = -n_tokens;
    
    std::vector<llama_token> tokens(n_tokens);
    n_tokens = llama_tokenize(vocab, prompt.c_str(), prompt.length(), tokens.data(), tokens.size(), true, true);
    
    if (n_tokens <= 0) {
        return makeJavaString(env, "Ошибка токенизации");
    }
    LOGI("Промпт разбит на %d токенов", n_tokens);

    if (n_tokens >= 4000) {
        return makeJavaString(env, "Промпт слишком длинный для контекста 4096.");
    }

    // 4. Очистка памяти и сэмплера
    llama_memory_clear(llama_get_memory(g_context), true);
    llama_sampler_reset(g_sampler);

    // 5. ДЕКОДИРОВАНИЕ ПРОМПТА ОДНИМ БАТЧЕМ (исправление ошибки -1)
    LOGI("Начинаем обработку промпта...");
    llama_batch batch = llama_batch_init(n_tokens, 0, 1);
    for (int i = 0; i < n_tokens; i++) {
        batch.token[i] = tokens[i];
        batch.pos[i] = i;
        batch.n_seq_id[i] = 1;
        batch.seq_id[i][0] = 0;
        batch.logits[i] = (i == n_tokens - 1); 
    }

    int res = llama_decode(g_context, batch);
    llama_batch_free(batch);

    if (res != 0) {
        LOGE("Ошибка llama_decode на промпте, код: %d", res);
        return makeJavaString(env, "Ошибка обработки промпта (decode " + std::to_string(res) + ")");
    }
    LOGI("Промпт успешно обработан.");

    // 6. ГЕНЕРАЦИЯ (по одному токену)
    std::string answer;
    int n_past = n_tokens;
    int limit = std::min((int)maxTokens, 512);

    for (int i = 0; i < limit; i++) {
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
        next.pos[0] = n_past++;
        next.n_seq_id[0] = 1;
        next.seq_id[0][0] = 0;
        next.logits[0] = true;

        res = llama_decode(g_context, next);
        llama_batch_free(next);
        
        if (res != 0) {
            LOGE("Ошибка llama_decode на шаге генерации %d, код: %d", i, res);
            break;
        }
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
