package sg.edu.ntu.scse.fyp.onnxwaifugenerator

import android.graphics.Bitmap
import android.graphics.Color
import android.util.Log
import androidx.compose.foundation.Image
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.selection.selectable
import androidx.compose.foundation.selection.selectableGroup
import androidx.compose.material.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.semantics.Role
import androidx.compose.ui.unit.dp
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import kotlin.math.roundToInt
import kotlin.random.Random

private const val TAG = "AppUi"

/**
 * Main UI
 */
@Composable
fun AppUi() {
    val resources = LocalContext.current.resources
    val scope = rememberCoroutineScope()
    val (onnxController, _) = remember { mutableStateOf(OnnxController(resources)) }

    val (isGenerating, setIsGenerating) = remember { mutableStateOf(false) }
    val (model, setModel) = remember { mutableStateOf(OnnxModel.SKYTNT) }
    val (seed, setSeed) = remember { mutableStateOf(0) }
    val (isRandomSeed, setRandomSeed) = remember { mutableStateOf(false) }
    val (trunc1, setTrunc1) = remember { mutableStateOf(1f) }
    val (trunc2, setTrunc2) = remember { mutableStateOf(1f) }
    val (noise, setNoise) = remember { mutableStateOf(0.5f) }

    val (image, setImage) = remember { mutableStateOf<Bitmap?>(null) }

    // Converts model output to Bitmap
    val modelToBitmap = { modelOutput: FloatArray, imgWidth: Int, imgHeight: Int ->
        val minVal = modelOutput.minOrNull() ?: -1.0f
        val maxVal = modelOutput.maxOrNull() ?: 1.0f
        val delta = maxVal - minVal

        val pixels = IntArray(imgWidth * imgHeight * 4)
        for (i in 0 until imgWidth * imgHeight) {
            pixels[i] = Color.rgb(
                ((modelOutput[i] - minVal) / delta * 255.0f).roundToInt(),
                ((modelOutput[i + imgWidth * imgHeight] - minVal) / delta * 255.0f).roundToInt(),
                ((modelOutput[i + 2 * imgWidth * imgHeight] - minVal) / delta * 255.0f).roundToInt()
            )
        }

        val bitmap = Bitmap.createBitmap(imgWidth, imgHeight, Bitmap.Config.ARGB_8888)
        bitmap.setPixels(pixels, 0, imgWidth, 0, 0, imgWidth, imgHeight)
        bitmap
    }

    val generateShape: () -> Unit = {
        setIsGenerating(true)

        val finalSeed: Int
        if (isRandomSeed) {
            finalSeed = Random.nextInt(0, Int.MAX_VALUE)
            setSeed(finalSeed)
        } else {
            finalSeed = seed
        }

        // Performs shape generation in a background thread
        scope.launch(Dispatchers.Default) {
            val (modelOutput, shape) = onnxController.generateImage(
                model, finalSeed, floatArrayOf(trunc1, trunc2), noise
            )

            Log.d(TAG, "generateShape: Output shape = ${shape.joinToString()}")
            val imgWidth = shape[3].toInt()
            val imgHeight = shape[2].toInt()

            val bitmap = modelToBitmap(modelOutput, imgWidth, imgHeight)

            withContext(Dispatchers.Main) {
                setImage(bitmap)
                setIsGenerating(false)
            }

            // Save to file in app storage
//        val fileName = "model-output-${System.currentTimeMillis()}.png"
//        val file = File(applicationContext.filesDir, fileName)
//        bitmap.compress(Bitmap.CompressFormat.PNG, 100, file.outputStream())
//        Log.d(
//            TAG,
//            "generateShape: Bitmap stored in ${applicationContext.filesDir.absolutePath}/$fileName"
//        )
        }
    }

    // Action(s) to perform when UI is destroyed
    DisposableEffect(onnxController) {
        onDispose {
            onnxController.close()
        }
    }

    // UI
    Column(
        Modifier
            .fillMaxSize()
            .padding(top = 16.dp, start = 16.dp, end = 16.dp)
    ) {
        Text("Model")
        Spacer(Modifier.padding(bottom = 8.dp))
        Column(Modifier.selectableGroup()) {
            OnnxModel.values().forEach { onnxModel ->
                Row(
                    Modifier.selectable(
                        selected = model == onnxModel,
                        onClick = { setModel(onnxModel) },
                        role = Role.RadioButton,
                        enabled = !isGenerating
                    )
                ) {

                    RadioButton(
                        selected = model == onnxModel,
                        onClick = null,
                        enabled = !isGenerating
                    )
                    Text(onnxModel.label)
                }
            }
        }
        Spacer(Modifier.padding(bottom = 16.dp))
        LabelledSlider(
            label = "Seed",
            value = seed.toFloat(),
            valueRange = 0f..Int.MAX_VALUE.toFloat(),
            onValueChange = { setSeed(it.toInt()) },
            isEnabled = !isGenerating && !isRandomSeed
        )
        Row(
            modifier = Modifier.align(Alignment.End),
            verticalAlignment = Alignment.CenterVertically
        ) {
            Checkbox(
                checked = isRandomSeed,
                onCheckedChange = setRandomSeed,
                enabled = !isGenerating
            )
            Text("Random")
        }
        ModelParamSlider(
            label = "Truncation 1",
            value = trunc1,
            maxValue = 2f,
            onValueChange = setTrunc1,
            isEnabled = !isGenerating
        )
        ModelParamSlider(
            label = "Truncation 2",
            value = trunc2,
            maxValue = 2f,
            onValueChange = setTrunc2,
            isEnabled = !isGenerating
        )
        ModelParamSlider(
            label = "Noise",
            value = noise,
            maxValue = 1f,
            onValueChange = setNoise,
            isEnabled = !isGenerating
        )
        Button(
            onClick = generateShape,
            enabled = !isGenerating
        ) {
            Text(text = "Generate")
        }
        if (isGenerating) {
            CircularProgressIndicator(Modifier.align(Alignment.CenterHorizontally))
        }
        if (image != null && !isGenerating) {
            Image(
                modifier = Modifier.align(Alignment.CenterHorizontally),
                bitmap = image.asImageBitmap(),
                contentDescription = ""
            )
        }
    }
}
