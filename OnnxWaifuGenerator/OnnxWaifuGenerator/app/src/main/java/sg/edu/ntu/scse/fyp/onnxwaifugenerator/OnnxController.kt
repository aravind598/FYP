package sg.edu.ntu.scse.fyp.onnxwaifugenerator

import ai.onnxruntime.OrtEnvironment
import android.content.res.Resources
import androidx.annotation.RawRes

class OnnxController(res: Resources) {
    private val skytntGenerator by lazy {
        val env = OrtEnvironment.getEnvironment()
        OnnxGenerator(
            env,
            env.createSession(loadModel(res, R.raw.g_mapping)),
            env.createSession(loadModel(res, R.raw.g_synthesis)),
            1024
        )
    }

    private val customGenerator by lazy {
        val env = OrtEnvironment.getEnvironment()
        OnnxGenerator(
            env,
            env.createSession(loadModel(res, R.raw.g_mapping_aravind)),
            env.createSession(loadModel(res, R.raw.g_synthesis_aravind)),
            512
        )
    }

    fun generateImage(
        model: OnnxModel,
        seed: Int,
        psi: FloatArray,
        noise: Float
    ): Pair<FloatArray, LongArray> {
        val generator = when (model) {
            OnnxModel.SKYTNT -> skytntGenerator
            OnnxModel.CUSTOM -> customGenerator
        }

        return generator.generateImage(seed, psi, noise)
    }

    fun close() {
        skytntGenerator.close()
        customGenerator.close()
    }

    /**
     * Reads model file from raw folder
     *
     * @param res Resources object from Android UI
     * @param rawId Resource ID of model file
     *
     * @return Model file as byte array
     */
    private fun loadModel(res: Resources, @RawRes rawId: Int): ByteArray {
        val modelInputStream = res.openRawResource(rawId)
        val modelBuffer = ByteArray(modelInputStream.available())
        modelInputStream.read(modelBuffer)
        return modelBuffer
    }
}