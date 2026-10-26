
package com.ramzes.ai

import android.app.Activity
import android.content.Context
import android.content.Intent
import android.net.Uri
import android.os.Bundle
import android.view.Gravity
import android.view.inputmethod.InputMethodManager
import android.widget.Button
import android.widget.EditText
import android.widget.LinearLayout
import android.widget.ScrollView
import android.widget.TextView
import org.json.JSONArray
import org.json.JSONObject
import java.io.File
import java.io.FileOutputStream
import kotlin.concurrent.thread

class MainActivity : Activity() {

    private lateinit var status: TextView
    private lateinit var chat: TextView
    private lateinit var input: EditText
    private lateinit var sendButton: Button
    private lateinit var modelButton: Button
    private lateinit var scrollView: ScrollView

    private val messages =
        mutableListOf<Pair<String, String>>()

    private val prefs by lazy {
        getSharedPreferences("ramzes_history", MODE_PRIVATE)
    }

    @Volatile
    private var modelLoaded = false

    @Volatile
    private var busy = false

    companion object {
        private const val PICK_MODEL = 100

        init {
            System.loadLibrary("ramzes_native")
        }
    }

    private external fun nativeLoadModel(
        path: String,
        contextSize: Int
    ): String

    private external fun nativeGenerate(
        prompt: String,
        maxTokens: Int
    ): String

    private external fun nativeUnloadModel()

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        createInterface()
        restoreHistory()

        status.text = "Модель не загружена"
    }

    private fun createInterface() {
        val root = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(16, 16, 16, 12)
        }

        val title = TextView(this).apply {
            text = "RAMZES AI"
            textSize = 26f
            gravity = Gravity.CENTER_VERTICAL
        }

        status = TextView(this).apply {
            textSize = 14f
            setPadding(0, 8, 0, 8)
        }

        modelButton = Button(this).apply {
            text = "Выбрать GGUF-модель"
            setOnClickListener {
                chooseModel()
            }
        }

        chat = TextView(this).apply {
            textSize = 16f
            setTextIsSelectable(true)
            setPadding(4, 8, 4, 8)
        }

        scrollView = ScrollView(this).apply {
            addView(chat)
        }

        input = EditText(this).apply {
            hint = "Напишите сообщение..."
            maxLines = 4
        }

        sendButton = Button(this).apply {
            text = "Отправить"
            setOnClickListener {
                sendMessage()
            }
        }

        val bottom = LinearLayout(this).apply {
            orientation = LinearLayout.HORIZONTAL
            gravity = Gravity.CENTER_VERTICAL

            addView(
                input,
                LinearLayout.LayoutParams(0, -2, 1f)
            )

            addView(sendButton)
        }

        root.addView(title)
        root.addView(status)
        root.addView(modelButton)

        root.addView(
            scrollView,
            LinearLayout.LayoutParams(-1, 0, 1f)
        )

        root.addView(bottom)

        setContentView(root)
    }

    private fun chooseModel() {
        if (busy) return

        val intent = Intent(Intent.ACTION_OPEN_DOCUMENT).apply {
            addCategory(Intent.CATEGORY_OPENABLE)
            type = "*/*"
        }

        startActivityForResult(intent, PICK_MODEL)
    }

    @Deprecated("Compatibility with Android document picker")
    override fun onActivityResult(
        requestCode: Int,
        resultCode: Int,
        data: Intent?
    ) {
        super.onActivityResult(requestCode, resultCode, data)

        if (requestCode != PICK_MODEL ||
            resultCode != RESULT_OK
        ) {
            return
        }

        val uri: Uri = data?.data ?: return

        busy = true
        sendButton.isEnabled = false
        modelButton.isEnabled = false
        status.text = "Импорт GGUF-модели..."

        thread {
            try {
                val modelFile = copyModel(uri)

                runOnUiThread {
                    status.text = "Загрузка модели..."
                }

                val result = nativeLoadModel(
                    modelFile.absolutePath,
                    2048
                )

                runOnUiThread {
                    modelLoaded = result == "OK"

                    status.text = if (modelLoaded) {
                        "Модель загружена: ${modelFile.name}"
                    } else {
                        result.removePrefix("ERROR: ")
                    }

                    busy = false
                    sendButton.isEnabled = true
                    modelButton.isEnabled = true
                }
            } catch (e: Exception) {
                runOnUiThread {
                    modelLoaded = false
                    busy = false

                    status.text =
                        "Ошибка загрузки: ${e.message ?: "неизвестная ошибка"}"

                    sendButton.isEnabled = true
                    modelButton.isEnabled = true
                }
            }
        }
    }

    private fun copyModel(uri: Uri): File {
        val directory = File(filesDir, "models")

        if (!directory.exists() && !directory.mkdirs()) {
            error("Не удалось создать папку моделей")
        }

        val destination = File(directory, "ramzes_model.gguf")
        val temporary = File(directory, "ramzes_model.tmp")

        try {
            val source = contentResolver.openInputStream(uri)
                ?: error("Не удалось открыть выбранный файл")

            source.use { inputStream ->
                FileOutputStream(temporary).use { outputStream ->
                    inputStream.copyTo(outputStream)
                }
            }

            require(temporary.length() > 0L) {
                "Выбран пустой файл"
            }

            if (destination.exists() && !destination.delete()) {
                error("Не удалось заменить предыдущую модель")
            }

            if (!temporary.renameTo(destination)) {
                temporary.copyTo(destination, overwrite = true)
                temporary.delete()
            }

            return destination
        } catch (e: Exception) {
            temporary.delete()
            throw e
        }
    }

    private fun sendMessage() {
        val question = input.text.toString().trim()

        if (question.isEmpty()) return

        if (busy) {
            status.text = "Дождитесь завершения текущей операции."
            return
        }

        if (!modelLoaded) {
            status.text = "Сначала загрузите GGUF-модель."
            return
        }

        input.setText("")
        messages.add("Вы" to question)
        renderChat()
        saveHistory()

        busy = true
        sendButton.isEnabled = false
        modelButton.isEnabled = false
        status.text = "RAMZES AI генерирует ответ..."

        val prompt = buildString {
            append(
                "You are RAMZES AI, a helpful assistant. " +
                    "Answer in the same language as the user.\n"
            )

            messages.takeLast(12).forEach { (speaker, text) ->
                if (speaker == "Вы") {
                    append("User: ")
                } else {
                    append("Assistant: ")
                }

                append(text)
                append('\n')
            }

            append("Assistant:")
        }

        (getSystemService(Context.INPUT_METHOD_SERVICE)
            as InputMethodManager)
            .hideSoftInputFromWindow(input.windowToken, 0)

        thread {
            val answer = try {
                nativeGenerate(prompt, 256).ifBlank {
                    "Модель не вернула текст. Попробуйте другой запрос."
                }
            } catch (e: Exception) {
                "Ошибка генерации: ${e.message ?: "неизвестная ошибка"}"
            }

            runOnUiThread {
                messages.add("RAMZES AI" to answer)

                renderChat()
                saveHistory()

                status.text = "Готово"
                busy = false
                sendButton.isEnabled = true
                modelButton.isEnabled = true
            }
        }
    }

    private fun renderChat() {
        chat.text = messages.joinToString("\n\n") {
            "${it.first}:\n${it.second}"
        }

        scrollView.post {
            scrollView.fullScroll(ScrollView.FOCUS_DOWN)
        }
    }

    private fun saveHistory() {
        val array = JSONArray()

        messages.forEach { (speaker, message) ->
            array.put(
                JSONObject()
                    .put("speaker", speaker)
                    .put("message", message)
            )
        }

        prefs.edit()
            .putString("messages", array.toString())
            .apply()
    }

    private fun restoreHistory() {
        try {
            val array = JSONArray(
                prefs.getString("messages", "[]")
            )

            for (i in 0 until array.length()) {
                val item = array.getJSONObject(i)

                messages.add(
                    item.getString("speaker") to
                        item.getString("message")
                )
            }

            renderChat()
        } catch (_: Exception) {
            messages.clear()
        }
    }

    override fun onDestroy() {
        // Не освобождаем нативную модель одновременно с активным вызовом.
        // Для полноценной версии жизненный цикл движка лучше вынести в сервис.
        if (!busy) {
            try {
                nativeUnloadModel()
            } catch (_: Exception) {
                // Библиотека могла не загрузиться.
            }
        }

        super.onDestroy()
    }
}
