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
import java.io.IOException
import kotlin.concurrent.thread

class MainActivity : Activity() {

    private lateinit var status: TextView
    private lateinit var chat: TextView
    private lateinit var input: EditText
    private lateinit var sendButton: Button
    private lateinit var modelButton: Button
    private lateinit var scrollView: ScrollView

    private val messages = mutableListOf<Pair<String, String>>()

    private val prefs by lazy {
        getSharedPreferences("ramzes_history", MODE_PRIVATE)
    }

    @Volatile
    private var modelLoaded = false

    @Volatile
    private var busy = false

    @Volatile
    private var activityDestroyed = false

    companion object {
        private const val PICK_MODEL = 100
        private const val MAX_HISTORY_MESSAGES = 80
        private const val PROMPT_HISTORY_MESSAGES = 6
        private const val CONTEXT_SIZE = 1024
        private const val MAX_GENERATION_TOKENS = 64

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

        activityDestroyed = false
        createInterface()
        restoreHistory()

        status.text = "Модель не загружена"
    }

    private fun createInterface() {
        val root = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(16, 12, 16, 12)
        }

        val title = TextView(this).apply {
            text = "RAMZES AI"
            textSize = 26f
            gravity = Gravity.CENTER_VERTICAL
            setPadding(0, 4, 0, 4)
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
            isFillViewport = true
            addView(
                chat,
                ScrollView.LayoutParams(
                    ScrollView.LayoutParams.MATCH_PARENT,
                    ScrollView.LayoutParams.WRAP_CONTENT
                )
            )
        }

        input = EditText(this).apply {
            hint = "Напишите сообщение..."
            maxLines = 4
            setSingleLine(false)
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

            addView(
                sendButton,
                LinearLayout.LayoutParams(-2, -2)
            )
        }

        root.addView(title)
        root.addView(status)
        root.addView(modelButton)

        root.addView(
            scrollView,
            LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.MATCH_PARENT,
                0,
                1f
            )
        )

        root.addView(bottom)
        setContentView(root)
    }

    private fun chooseModel() {
        if (busy) {
            showStatus("Дождитесь завершения текущей операции.")
            return
        }

        val intent = Intent(Intent.ACTION_OPEN_DOCUMENT).apply {
            addCategory(Intent.CATEGORY_OPENABLE)
            type = "*/*"
        }

        try {
            @Suppress("DEPRECATION")
            startActivityForResult(intent, PICK_MODEL)
        } catch (e: Exception) {
            showStatus(
                "Не удалось открыть выбор файла: ${errorMessage(e)}"
            )
        }
    }

    @Deprecated("Используется для совместимости с существующим проектом")
    override fun onActivityResult(
        requestCode: Int,
        resultCode: Int,
        data: Intent?
    ) {
        super.onActivityResult(requestCode, resultCode, data)

        if (requestCode != PICK_MODEL || resultCode != RESULT_OK) {
            return
        }

        val uri: Uri = data?.data ?: run {
            showStatus("Файл не выбран.")
            return
        }

        if (busy) {
            showStatus("Дождитесь завершения текущей операции.")
            return
        }

        busy = true
        modelLoaded = false
        setControlsEnabled(false)
        showStatus("Импорт GGUF-модели...")

        thread(name = "ramzes-model-loader") {
            var loaded = false

            try {
                val modelFile = copyModel(uri)

                if (activityDestroyed) {
                    return@thread
                }

                runOnUiThread {
                    if (!activityDestroyed) {
                        status.text = "Загрузка модели в память..."
                    }
                }

                val result = nativeLoadModel(
                    modelFile.absolutePath,
                    CONTEXT_SIZE
                )

                loaded = result == "OK"

                runOnUiThread {
                    if (!activityDestroyed) {
                        modelLoaded = loaded

                        status.text = if (loaded) {
                            "Модель загружена: ${modelFile.name}"
                        } else {
                            result.removePrefix("ERROR: ")
                        }

                        setControlsEnabled(true)
                    }
                }
            } catch (e: Exception) {
                runOnUiThread {
                    if (!activityDestroyed) {
                        modelLoaded = false

                        status.text =
                            "Ошибка загрузки: ${errorMessage(e)}"

                        setControlsEnabled(true)
                    }
                }
            } finally {
                busy = false
            }
        }
    }

    /**
     * Копирует модель сначала во временный файл.
     * Существующая модель на диске не удаляется,
     * пока новая не скопирована успешно.
     */
    private fun copyModel(uri: Uri): File {
        val directory = File(filesDir, "models")

        if (!directory.exists() && !directory.mkdirs()) {
            throw IOException("Не удалось создать папку моделей.")
        }

        val destination = File(directory, "ramzes_model.gguf")
        val temporary = File(directory, "ramzes_model.tmp")

        try {
            val source = contentResolver.openInputStream(uri)
                ?: throw IOException("Не удалось открыть выбранный файл.")

            source.use { inputStream ->
                FileOutputStream(temporary).use { outputStream ->
                    inputStream.copyTo(outputStream)
                    outputStream.fd.sync()
                }
            }

            if (!temporary.isFile || temporary.length() == 0L) {
                throw IOException("Выбран пустой файл.")
            }

            // Проверка расширения помогает избежать выбора случайного файла.
            // Фактическая проверка формата GGUF выполняется нативной библиотекой.
            val displayName = queryDisplayName(uri)
            if (
                displayName != null &&
                !displayName.endsWith(".gguf", ignoreCase = true)
            ) {
                throw IOException("Выберите файл модели с расширением .gguf.")
            }

            // Замена выполняется после успешного копирования.
            // Сохраняем резервную копию на случай ошибки переименования.
            val backup = File(directory, "ramzes_model.bak")
            var backupCreated = false

            if (destination.exists()) {
                if (backup.exists() && !backup.delete()) {
                    throw IOException("Не удалось удалить старую резервную копию.")
                }

                if (!destination.renameTo(backup)) {
                    throw IOException("Не удалось сохранить предыдущую модель.")
                }

                backupCreated = true
            }

            val installed = temporary.renameTo(destination)

            if (!installed) {
                try {
                    temporary.copyTo(destination, overwrite = true)
                    temporary.delete()
                } catch (e: Exception) {
                    destination.delete()

                    if (backupCreated) {
                        backup.renameTo(destination)
                    }

                    throw IOException(
                        "Не удалось сохранить новую модель.",
                        e
                    )
                }
            }

            if (backupCreated) {
                backup.delete()
            }

            return destination
        } catch (e: Exception) {
            temporary.delete()
            throw e
        }
    }

    private fun queryDisplayName(uri: Uri): String? {
        return try {
            contentResolver.query(
                uri,
                arrayOf(android.provider.OpenableColumns.DISPLAY_NAME),
                null,
                null,
                null
            )?.use { cursor ->
                if (cursor.moveToFirst()) {
                    val index = cursor.getColumnIndex(
                        android.provider.OpenableColumns.DISPLAY_NAME
                    )

                    if (index >= 0) cursor.getString(index) else null
                } else {
                    null
                }
            }
        } catch (_: Exception) {
            null
        }
    }

    private fun buildPrompt(): String {
        val recentMessages = messages.takeLast(PROMPT_HISTORY_MESSAGES)

        return buildString {
            append(
                "<|im_start|>system\n" +
                    "Ты RAMZES AI, полезный и точный помощник. " +
                    "Отвечай на языке пользователя. " +
                    "Если пользователь пишет по-русски, отвечай по-русски.\n" +
                    "<|im_end|>\n"
            )

            recentMessages.forEach { (speaker, message) ->
                val role = if (speaker == "Вы") "user" else "assistant"

                append("<|im_start|>")
                append(role)
                append('\n')
                append(message)
                append("\n<|im_end|>\n")
            }

            append("<|im_start|>assistant\n")
        }
    }

    private fun sendMessage() {
        val question = input.text.toString().trim()

        if (question.isEmpty()) {
            return
        }

        if (busy) {
            showStatus("Дождитесь завершения текущей операции.")
            return
        }

        if (!modelLoaded) {
            showStatus("Сначала загрузите GGUF-модель.")
            return
        }

        input.setText("")
        hideKeyboard()

        messages.add("Вы" to question)
        trimHistory()
        renderChat()
        saveHistory()

        val prompt = buildPrompt()

        busy = true
        setControlsEnabled(false)
        showStatus("RAMZES AI генерирует ответ...")

        thread(name = "ramzes-generation") {
            val answer = try {
                nativeGenerate(prompt, MAX_GENERATION_TOKENS).trim()
                    .ifBlank {
                        "Модель не вернула текст. Проверьте совместимость модели и шаблона чата."
                    }
            } catch (e: Exception) {
                "Ошибка генерации: ${errorMessage(e)}"
            }

            runOnUiThread {
                if (!activityDestroyed) {
                    messages.add("RAMZES AI" to answer)
                    trimHistory()
                    renderChat()
                    saveHistory()

                    status.text = if (modelLoaded) {
                        "Готово"
                    } else {
                        "Модель не загружена"
                    }

                    setControlsEnabled(true)
                }
            }

            busy = false
        }
    }

    private fun trimHistory() {
        while (messages.size > MAX_HISTORY_MESSAGES) {
            messages.removeAt(0)
        }
    }

    private fun renderChat() {
        chat.text = messages.joinToString("\n\n") { (speaker, message) ->
            "$speaker:\n$message"
        }

        scrollView.post {
            scrollView.fullScroll(ScrollView.FOCUS_DOWN)
        }
    }

    private fun saveHistory() {
        try {
            val array = JSONArray()

            messages.takeLast(MAX_HISTORY_MESSAGES).forEach {
                (speaker, message) ->
                array.put(
                    JSONObject()
                        .put("speaker", speaker)
                        .put("message", message)
                )
            }

            prefs.edit()
                .putString("messages", array.toString())
                .apply()
        } catch (_: Exception) {
            showStatus("Не удалось сохранить историю чата.")
        }
    }

    private fun restoreHistory() {
        try {
            val json = prefs.getString("messages", "[]") ?: "[]"
            val array = JSONArray(json)

            messages.clear()

            val start = maxOf(0, array.length() - MAX_HISTORY_MESSAGES)

            for (i in start until array.length()) {
                val item = array.optJSONObject(i) ?: continue
                val speaker = item.optString("speaker", "")
                val message = item.optString("message", "")

                if (
                    speaker.isNotBlank() &&
                    message.isNotBlank()
                ) {
                    messages.add(speaker to message)
                }
            }

            renderChat()
        } catch (_: Exception) {
            messages.clear()
            renderChat()
        }
    }

    private fun hideKeyboard() {
        try {
            val manager = getSystemService(
                Context.INPUT_METHOD_SERVICE
            ) as InputMethodManager

            manager.hideSoftInputFromWindow(input.windowToken, 0)
        } catch (_: Exception) {
            // Закрытие клавиатуры не должно прерывать отправку сообщения.
        }
    }

    private fun showStatus(message: String) {
        if (!activityDestroyed) {
            status.text = message
        }
    }

    private fun setControlsEnabled(enabled: Boolean) {
        if (activityDestroyed) return

        sendButton.isEnabled = enabled
        modelButton.isEnabled = enabled
        input.isEnabled = enabled
    }

    private fun errorMessage(e: Exception): String {
        return e.message?.takeIf { it.isNotBlank() }
            ?: e.javaClass.simpleName
    }

    override fun onDestroy() {
        activityDestroyed = true

        // Нативная функция защищена mutex в C++.
        // Выгрузка запускается вне главного UI-потока.
        thread(name = "ramzes-model-cleanup") {
            try {
                nativeUnloadModel()
            } catch (_: Exception) {
                // При закрытии Activity ошибка очистки не должна приводить к сбою.
            }
        }

        super.onDestroy()
    }
}
