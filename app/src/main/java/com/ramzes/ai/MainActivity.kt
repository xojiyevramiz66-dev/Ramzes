package com.ramzes.ai

import android.content.Context
import android.net.Uri
import android.os.Bundle
import android.view.Gravity
import android.view.inputmethod.InputMethodManager
import android.widget.Button
import android.widget.EditText
import android.widget.LinearLayout
import android.widget.ScrollView
import android.widget.TextView
import androidx.activity.ComponentActivity
import androidx.activity.result.contract.ActivityResultContracts
import org.json.JSONArray
import org.json.JSONObject
import java.io.File
import java.io.FileOutputStream
import java.io.IOException
import java.util.concurrent.atomic.AtomicReference
import kotlin.concurrent.thread

class MainActivity : ComponentActivity() {
    private lateinit var status: TextView
    private lateinit var chat: TextView
    private lateinit var input: EditText
    private lateinit var sendButton: Button
    private lateinit var modelButton: Button
    private lateinit var scroll: ScrollView

    private val messages = mutableListOf<Pair<String, String>>()

    @Volatile private var modelLoaded = false
    @Volatile private var busy = false
    @Volatile private var destroyed = false

    // Поток для генерации
    private var generateThread: Thread? = null

    private val prefs by lazy {
        getSharedPreferences("ramzes_history", MODE_PRIVATE)
    }

    companion object {
        private const val CONTEXT_SIZE = 512 // Уменьшено для стабильности на телефонах
        private const val MAX_TOKENS = 64     // Уменьшено, чтобы избежать долгого зависания
        private const val MAX_HISTORY = 20
        private const val PROMPT_HISTORY = 4  // Меньше истории = меньше нагрузка на контекст
        private const val GENERATION_TIMEOUT_MS = 30000L // 30 секунд максимум

        init {
            try {
                System.loadLibrary("ramzes_native")
            } catch (e: UnsatisfiedLinkError) {
                e.printStackTrace()
            }
        }
    }

    private external fun nativeLoadModel(path: String, contextSize: Int): String
    private external fun nativeGenerate(prompt: String, maxTokens: Int): String
    private external fun nativeUnloadModel()

    private val pickModelLauncher = registerForActivityResult(
        ActivityResultContracts.OpenDocument()
    ) { uri: Uri? ->
        if (uri != null) {
            loadModelFromUri(uri)
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        createUi()
        restoreHistory()

        status.text = if (modelLoaded) "Модель загружена" else "Выберите GGUF-модель"
        setControls(!busy)
    }

    private fun createUi() {
        val root = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(16, 12, 16, 12)
        }

        root.addView(TextView(this).apply {
            text = "RAMZES AI"
            textSize = 26f
            gravity = Gravity.CENTER_VERTICAL
        })

        status = TextView(this).apply {
            textSize = 14f
            setPadding(0, 8, 0, 8)
        }
        root.addView(status)

        modelButton = Button(this).apply {
            text = "Выбрать GGUF-модель"
            setOnClickListener { chooseModel() }
        }
        root.addView(modelButton)

        chat = TextView(this).apply {
            textSize = 16f
            setTextIsSelectable(true)
            setPadding(4, 8, 4, 8)
        }

        scroll = ScrollView(this).apply {
            isFillViewport = true
            addView(chat)
        }

        root.addView(scroll, LinearLayout.LayoutParams(-1, 0, 1f))

        input = EditText(this).apply {
            hint = "Напишите сообщение..."
            maxLines = 4
        }

        sendButton = Button(this).apply {
            text = "Отправить"
            setOnClickListener { sendMessage() }
        }

        val bottom = LinearLayout(this).apply {
            orientation = LinearLayout.HORIZONTAL
            addView(input, LinearLayout.LayoutParams(0, -2, 1f))
            addView(sendButton)
        }
        root.addView(bottom)
        setContentView(root)
    }

    private fun chooseModel() {
        if (busy) return
        try {
            pickModelLauncher.launch(arrayOf("*/*"))
        } catch (e: Exception) {
            status.text = "Не удалось открыть файл: ${e.message}"
        }
    }

    private fun loadModelFromUri(uri: Uri) {
        if (busy) return
        busy = true
        modelLoaded = false
        setControls(false)
        status.text = "Копирование модели..."

        thread(name = "ramzes-load") {
            var result = "Ошибка загрузки модели"

            try {
                val file = copyModel(uri)
                if (destroyed) return@thread

                runOnUiThread {
                    if (!destroyed) status.text = "Загрузка модели в память..."
                }

                result = nativeLoadModel(file.absolutePath, CONTEXT_SIZE)
            } catch (e: Throwable) {
                result = "Ошибка: ${e.message ?: e.javaClass.simpleName}"
                e.printStackTrace()
            }

            val finalResult = result
            runOnUiThread {
                if (!destroyed) {
                    modelLoaded = finalResult == "OK"
                    status.text = if (modelLoaded) "Модель загружена" else finalResult
                    setControls(true)
                }
                busy = false
            }
        }
    }

    private fun copyModel(uri: Uri): File {
        val dir = File(filesDir, "models")
        if (!dir.exists() && !dir.mkdirs()) {
            throw IOException("Не удалось создать папку моделей")
        }

        val name = queryDisplayName(uri)
        if (name != null && !name.endsWith(".gguf", true)) {
            throw IOException("Выберите файл .gguf")
        }

        val destination = File(dir, "ramzes_model.gguf")
        val temp = File(dir, "ramzes_model.tmp")

        contentResolver.openInputStream(uri).use { source ->
            if (source == null) throw IOException("Не удалось открыть файл")
            FileOutputStream(temp).use { output ->
                source.copyTo(output)
                output.fd.sync()
            }
        }

        if (!temp.exists() || temp.length() == 0L) {
            temp.delete()
            throw IOException("Файл модели пуст")
        }

        if (destination.exists() && !destination.delete()) {
            temp.delete()
            throw IOException("Не удалось заменить старую модель")
        }

        if (!temp.renameTo(destination)) {
            temp.copyTo(destination, overwrite = true)
            temp.delete()
        }

        return destination
    }

    private fun queryDisplayName(uri: Uri): String? {
        return try {
            contentResolver.query(
                uri,
                arrayOf(android.provider.OpenableColumns.DISPLAY_NAME),
                null, null, null
            )?.use { cursor ->
                if (cursor.moveToFirst()) {
                    val i = cursor.getColumnIndex(
                        android.provider.OpenableColumns.DISPLAY_NAME
                    )
                    if (i >= 0) cursor.getString(i) else null
                } else null
            }
        } catch (_: Exception) {
            null
        }
    }

    private fun buildPrompt(): String {
        val recent = messages.takeLast(PROMPT_HISTORY)

        return buildString {
            append(
                "<|im_start|>system\n" +
                "Ты RAMZES AI, полезный помощник. " +
                "Отвечай на языке пользователя.\n<|im_end|>\n"
            )

            recent.forEach { (speaker, text) ->
                val role = if (speaker == "Вы") "user" else "assistant"
                append("<|im_start|>$role\n$text\n<|im_end|>\n")
            }

            append("<|im_start|>assistant\n")
        }
    }

    private fun sendMessage() {
        val question = input.text.toString().trim()
        if (question.isEmpty() || busy) return

        if (!modelLoaded) {
            status.text = "Сначала загрузите модель"
            return
        }

        input.setText("")
        try {
            val manager = getSystemService(Context.INPUT_METHOD_SERVICE) as InputMethodManager
            manager.hideSoftInputFromWindow(input.windowToken, 0)
        } catch (_: Exception) {}

        messages.add("Вы" to question)
        trimHistory()
        renderChat()
        saveHistory()

        val prompt = buildPrompt()
        busy = true
        setControls(false)
        status.text = "RAMZES AI генерирует ответ..."

        val resultRef = AtomicReference<String>("Превышено время ожидания (30 сек). Модель зависла или не хватает памяти.")

        generateThread = thread(name = "ramzes-generate") {
            // Запускаем нативную генерацию в отдельном потоке
            val worker = thread {
                try {
                    android.util.Log.d("RAMZES", "Начало генерации. Длина промпта: ${prompt.length}")
                    val result = nativeGenerate(prompt, MAX_TOKENS).trim()
                    android.util.Log.d("RAMZES", "Генерация завершена. Длина ответа: ${result.length}")
                    
                    if (result.isBlank()) {
                        resultRef.set("Модель вернула пустой ответ. Проверьте Logcat.")
                    } else {
                        resultRef.set(result)
                    }
                } catch (e: Throwable) {
                    resultRef.set("Ошибка генерации: ${e.message ?: e.javaClass.simpleName}")
                    e.printStackTrace()
                }
            }

            // Ждем завершения максимум 30 секунд
            worker.join(GENERATION_TIMEOUT_MS)

            if (worker.isAlive) {
                android.util.Log.e("RAMZES", "Таймаут генерации! Модель зависла.")
                worker.interrupt()
                // Примечание: нативный C++ код не прервется, но UI мы разблокируем
            }

            val finalAnswer = resultRef.get()

            runOnUiThread {
                if (!destroyed) {
                    messages.add("RAMZES AI" to finalAnswer)
                    trimHistory()
                    renderChat()
                    saveHistory()
                    status.text = if (finalAnswer.contains("Превышено время")) "Таймаут" else "Готово"
                    setControls(true)
                }
                busy = false
                generateThread = null
            }
        }
    }

    private fun trimHistory() {
        while (messages.size > MAX_HISTORY) messages.removeAt(0)
    }

    private fun renderChat() {
        chat.text = messages.joinToString("\n\n") { (speaker, text) ->
            "$speaker:\n$text"
        }
        scroll.post { scroll.fullScroll(ScrollView.FOCUS_DOWN) }
    }

    private fun saveHistory() {
        try {
            val array = JSONArray()
            messages.forEach { (speaker, text) ->
                array.put(
                    JSONObject()
                        .put("speaker", speaker)
                        .put("message", text)
                )
            }
            prefs.edit().putString("messages", array.toString()).apply()
        } catch (_: Exception) {}
    }

    private fun restoreHistory() {
        try {
            val array = JSONArray(prefs.getString("messages", "[]") ?: "[]")
            for (i in 0 until array.length()) {
                val item = array.optJSONObject(i) ?: continue
                val speaker = item.optString("speaker")
                val text = item.optString("message")
                if (speaker.isNotBlank() && text.isNotBlank()) {
                    messages.add(speaker to text)
                }
            }
            trimHistory()
        } catch (_: Exception) {
            messages.clear()
        }
        renderChat()
    }

    private fun setControls(enabled: Boolean) {
        if (destroyed) return
        sendButton.isEnabled = enabled
        modelButton.isEnabled = enabled
        input.isEnabled = enabled
    }

    override fun onDestroy() {
        destroyed = true

        val t = generateThread
        if (t != null && t.isAlive) {
            try {
                t.join(2000)
            } catch (_: InterruptedException) {}
        }

        try {
            nativeUnloadModel()
        } catch (_: Throwable) {}

        super.onDestroy()
    }
}
