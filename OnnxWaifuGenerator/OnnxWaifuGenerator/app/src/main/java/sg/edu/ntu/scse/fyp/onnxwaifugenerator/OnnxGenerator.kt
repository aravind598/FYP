package sg.edu.ntu.scse.fyp.onnxwaifugenerator

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.util.Log
import java.nio.FloatBuffer
import kotlin.random.Random

private const val TAG = "OnnxGenerator"

/**
 * This class contains Onnx-related code to generate images from given models
 *
 * @param env Onnx environment
 * @param mappingFileId Raw resource ID of mapping model
 * @param synthesisFileId Raw resource ID of synthesis model
 */
class OnnxGenerator(
    private val env: OrtEnvironment,
    private val mappingSession: OrtSession,
    private val synthesisSession: OrtSession,
    private val zSize: Int
) {
    /**
     * Generates an image using models
     *
     * @param seed Seed used for random number generator
     * @param psi Array of slider values
     * @param noise Noise value for synthesis
     *
     * @return Output from mapping & synthesis models as float array and shape of output Tensor as
     * a long array
     */
    fun generateImage(seed: Int, psi: FloatArray, noise: Float): Pair<FloatArray, LongArray> {
        Log.d(TAG, "++generateImage++")
        val startTime = System.currentTimeMillis()
        val ran = Random(seed)

        // Run mapping
        val zBuffer = FloatBuffer.wrap(FloatArray(zSize) { ran.nextFloat() })
        val zTensor = OnnxTensor.createTensor(env, zBuffer, longArrayOf(1, zSize.toLong()))

        val psiBuffer = FloatBuffer.wrap(psi)
        val psiTensor = OnnxTensor.createTensor(env, psiBuffer, longArrayOf(psi.size.toLong()))

        val mappingInputs = mapOf(
            mappingSession.inputNames.elementAt(0) to zTensor,
            mappingSession.inputNames.elementAt(1) to psiTensor
        )

        val mappingOutput = mappingSession.run(mappingInputs)

        // Run synthesis
        val noiseBuffer = FloatBuffer.wrap(floatArrayOf(noise))
        val noiseTensor = OnnxTensor.createTensor(env, noiseBuffer, longArrayOf(1))

        val synthesisInputs = mapOf(
            synthesisSession.inputNames.elementAt(0) to mappingOutput.get(0) as OnnxTensor,
            synthesisSession.inputNames.elementAt(1) to noiseTensor
        )

        val synthesisOutput = synthesisSession.run(synthesisInputs).get(0) as OnnxTensor
        val outputInfo = synthesisOutput.info
        Log.d(TAG, "generateImage: Synthesis output info = $outputInfo")
        Log.d(TAG, "--generateImage--   Time taken to run = ${System.currentTimeMillis() - startTime}ms")
        return synthesisOutput.floatBuffer.array() to outputInfo.shape
    }

    /**
     * Closes sessions so that they are cleared from memory
     */
    fun close() {
        mappingSession.close()
        synthesisSession.close()
    }
}