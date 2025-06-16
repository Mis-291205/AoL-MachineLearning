package com.example.deepfakedetector

import android.graphics.Bitmap
import android.graphics.Color
import android.net.Uri
import android.os.Bundle
import android.widget.*
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import org.tensorflow.lite.Interpreter
import java.io.FileInputStream
import java.io.IOException
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel
import android.graphics.ImageDecoder
import android.provider.MediaStore
import android.content.pm.PackageManager
import android.graphics.BitmapFactory
import android.os.Build
import android.util.Log
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import androidx.core.graphics.scale
import androidx.core.graphics.get

class DetectionActivity : AppCompatActivity() {

    private lateinit var tflite: Interpreter
    private lateinit var imageView: ImageView
    private lateinit var resultText: TextView

    // Launcher untuk pilih gambar dari galeri
    private val galleryLauncher = registerForActivityResult(ActivityResultContracts.GetContent()) { uri: Uri? ->
        uri?.let {
            val bitmap = uriToBitmap(it)
            bitmap?.let { bmp -> predict(bmp) }
        }
    }

    // Launcher untuk ambil foto dari kamera (preview)
    private val cameraLauncher = registerForActivityResult(ActivityResultContracts.TakePicturePreview()) { bitmap: Bitmap? ->
        bitmap?.let { predict(it) }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_detection)

        if (ContextCompat.checkSelfPermission(this, android.Manifest.permission.CAMERA)
            != PackageManager.PERMISSION_GRANTED) {
            ActivityCompat.requestPermissions(
                this,
                arrayOf(android.Manifest.permission.CAMERA),
                100
            )
        }

        imageView = findViewById(R.id.image_view)
        resultText = findViewById(R.id.result_text)

        val btnUpload = findViewById<Button>(R.id.btn_upload)
        val btnCamera = findViewById<Button>(R.id.btn_camera)
//        val btnReset = findViewById<Button>(R.id.btn_reset)

        btnUpload.setOnClickListener {
            galleryLauncher.launch("image/*")
        }

        btnCamera.setOnClickListener {
            cameraLauncher.launch(null)
        }

//        btnReset.setOnClickListener {
//            imageView.setImageDrawable(null)
//            resultText.text = ""
//        }

        // Load model
        try {
            tflite = Interpreter(loadModelFile("best_model.tflite"))
            val inputTensor = tflite.getInputTensor(0)
            val outputTensor = tflite.getOutputTensor(0)

            Log.d("ModelShape", "Input shape: ${inputTensor.shape().contentToString()}, type: ${inputTensor.dataType()}")
            Log.d("ModelShape", "Output shape: ${outputTensor.shape().contentToString()}, type: ${outputTensor.dataType()}")
        } catch (e: Exception) {
            Toast.makeText(this, "Gagal memuat model: ${e.message}", Toast.LENGTH_LONG).show()
            e.printStackTrace()
        }

    }

    private fun loadModelFile(filename: String): MappedByteBuffer {
        val fileDescriptor = assets.openFd(filename)
        val inputStream = FileInputStream(fileDescriptor.fileDescriptor)
        val fileChannel = inputStream.channel
        val startOffset = fileDescriptor.startOffset
        val declaredLength = fileDescriptor.declaredLength
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength)
    }

    private fun uriToBitmap(uri: Uri): Bitmap? {
        return try {
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.P) {
                // Gunakan ImageDecoder untuk Android 9 (API 28) ke atas
                val source = ImageDecoder.createSource(contentResolver, uri)
                ImageDecoder.decodeBitmap(source)
            } else {
                // Gunakan openInputStream untuk Android 8 ke bawah
                val inputStream = contentResolver.openInputStream(uri)
                BitmapFactory.decodeStream(inputStream)
            }
        } catch (e: IOException) {
            e.printStackTrace()
            null
        }
    }


    private fun preprocessImage(bitmap: Bitmap): Array<Array<Array<FloatArray>>> {
        val convertedBitmap = bitmap.copy(Bitmap.Config.ARGB_8888, true)
        val resized = convertedBitmap.scale(224, 224)
        val input = Array(1) { Array(224) { Array(224) { FloatArray(3) } } }

        for (x in 0 until 224) {
            for (y in 0 until 224) {
                val pixel = resized[x, y]
                input[0][y][x][0] = Color.red(pixel) / 255.0f
                input[0][y][x][1] = Color.green(pixel) / 255.0f
                input[0][y][x][2] = Color.blue(pixel) / 255.0f
            }
        }
        return input
    }

    private fun predict(bitmap: Bitmap) {
        try {
            val input = preprocessImage(bitmap)
            val output = Array(49) { FloatArray(2) } // shape dari model output

            tflite.run(input, output)
            Log.d("ModelOutput", "First patch output: ${output[0].contentToString()}")

            // Ambil rata-rata dari semua 49 patch
            val avgOutput = FloatArray(2)
            for (i in 0 until 49) {
                avgOutput[0] += output[i][0]
                avgOutput[1] += output[i][1]
            }
            avgOutput[0] /= 49f
            avgOutput[1] /= 49f

            val predictedIndex = if (avgOutput[0] > avgOutput[1]) 0 else 1
            val labels = listOf("REAL", "FAKE")
            val result = labels[predictedIndex]

            imageView.setImageBitmap(bitmap)
            resultText.text = "Result: $result"
            resultText.setTextColor(if (result == "REAL") Color.GREEN else Color.RED)
        } catch (e: Exception) {
            e.printStackTrace()
            Toast.makeText(this, "Gagal prediksi: ${e.message}", Toast.LENGTH_LONG).show()
        }
    }


}
