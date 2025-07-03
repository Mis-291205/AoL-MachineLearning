package com.example.wastedetector

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
            tflite = Interpreter(loadModelFile("model.tflite"))
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

        val mean = floatArrayOf(103.939f, 116.779f, 123.68f)

        for (y in 0 until 224) {
            for (x in 0 until 224) {
                val pixel = resized.getPixel(x, y)
                val r = Color.red(pixel).toFloat()
                val g = Color.green(pixel).toFloat()
                val b = Color.blue(pixel).toFloat()

                // Convert RGB to BGR and subtract mean
                input[0][y][x][0] = b - mean[0]  // B
                input[0][y][x][1] = g - mean[1]  // G
                input[0][y][x][2] = r - mean[2]  // R
            }
        }
        return input
    }




    private fun predict(bitmap: Bitmap) {
        try {
            val input = preprocessImage(bitmap)
            val output = Array(1) { FloatArray(1) } // Output shape [1][1]

            tflite.run(input, output)
            val probRecyclable = output[0][0]
            val threshold = 0.5f

            val predictedIndex = if (probRecyclable > threshold) 1 else 0
            val labels = listOf("Organic", "Recyclable")
            val result = labels[predictedIndex]

            imageView.setImageBitmap(bitmap)
            resultText.text = "Result: $result"
            resultText.setTextColor(if (result == "Organic") Color.GREEN else Color.RED)
        } catch (e: Exception) {
            e.printStackTrace()
            Toast.makeText(this, "Gagal prediksi: ${e.message}", Toast.LENGTH_LONG).show()
        }
    }



}
