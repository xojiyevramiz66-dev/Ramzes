
package com.ramzes.ai

import android.app.Activity
import android.content.Intent
import android.net.Uri
import android.os.Bundle
import android.view.Gravity
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

    private val messages = mutableListOf<Pair<String, String>>()

    private val prefs by lazy {
        getSharedPreferences("ramzes_history", MODE_PRIVATE)
    }

    companion object {
        private const val PICK_MODEL = 100
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        createInterface()
        restoreHistory()
    }

    private fun createInterface() {
        val root = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(16, 16, 16, 12)
        }

        val title = TextView(this).apply {
            text = "Dumdam AI"
            textSize = 26f
            gravity = Gravity.CENTER_VERTICAL
        }

        status = TextView(this).apply {
            text = "Локальная модель не загружена"
            textSize = 14f
            setPadding(0, 8, 0, 8)
        }

        val modelButton = Button(this).apply {
            text = "Выбрать GGUF-модель"
            setOnClickListener { chooseModel() }
        }

        chat = TextView(this).apply {
            textSize = 16f
            setTextIsSelectable(true)
            setPadding(4, 8, 4, 8)
        }

        val scroll = ScrollView(this).apply {
            addView(chat)
        }

        input = EditText(this).apply {
            hint = "Напишите сообщение"
            maxLines = 4
        }

        sendButton = Button(this).apply {
            text = "Отправить"
            setOnClickListener { sendMessage() }
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
            scroll,
            LinearLayout.LayoutParams(-1, 0, 1f)
        )
        root.addView(bottom)

        setContentView(root)
    }

    private fun chooseModel() {
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

        if (requestCode != PICK_MODEL || resultCode != RESULT_OK) {
            return
        }

        val uri: Uri = data?.data ?: return

        status.text = "Импорт модели..."
        sendButton.isEnabled = false

        thread {
            try {
                val model = copyModel(uri)

                runOnUiThread {
                    status.text =
                        "Файл импортирован: ${model.name}. Движок ещё не подключён."
                    sendButton.isEnabled = true
                }
            } catch (e: Exception) {
                runOnUiThread {
                    status.text =
                        "Ошибка импорта: ${e.message ?: "неизвестная ошибка"}"
                    sendButton.isEnabled = true
                }
            }
        }
    }

    private fun copyModel(uri: Uri): File {
        val directory = File(filesDir, "models")
        if (!directory.exists()) {
            directory.mkdirs()
        }

        val file = File(directory, "model.gguf")

        val source = contentResolver.openInputStream(uri)
            ?: error("Не удалось открыть файл")

        source.use { inputStream ->
            FileOutputStream(file).use { outputStream ->
                inputStream.copyTo(outputStream)
            }
        }

        require(file.length() > 0) {
            "Выбран пустой файл"
        }

        return file
    }

    private fun sendMessage() {
        val text = input.text.toString().trim()
        if (text.isEmpty()) return

        messages.add("Вы" to text)
        input.setText("")
        renderChat()
        saveHistory()

        // Здесь позже будет вызов локальной модели llama.cpp.
        messages.add(
            "Ramzes AI" to
                "Интерфейс работает. Для генерации ответа необходимо подключить движок llama.cpp."
        )

        renderChat()
        saveHistory()
    }

    private fun renderChat() {
        chat.text = messages.joinToString("\n\n") {
            "${it.first}:\n${it.second}"
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
}
