import numpy as np
import os
import pickle
import streamlit as st
st.set_page_config(
    page_title="Ex-stream-ly Cool App",
    page_icon="🧊",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://www.extremelycoolapp.com/help',
        'Report a bug': "https://www.extremelycoolapp.com/bug",
        'About': "# This is a header. This is an *extremely* cool app!"
    }
)
import sys
import onnxruntime as rt
import numpy as np
import PIL
import datetime
sys.path.append("models")
sys.path.append("pg_gan")




st.title("Aravind's FYP Demo")
"""This demo demonstrates  using [Nvidia's Progressive Growing of GANs](https://research.nvidia.com/publication/2017-10_Progressive-Growing-of) and 
Shaobo Guan's [Transparent Latent-space GAN method](https://blog.insightdatascience.com/generating-custom-photo-realistic-faces-using-ai-d170b1b59255) 
for tuning the output face's characteristics. For more information, check out the tutorial on [Towards Data Science](https://towardsdatascience.com/building-machine-learning-apps-with-streamlit-667cef3ff509)."""

# Download all data files if they aren't already in the working directory.
# for filename in EXTERNAL_DEPENDENCIES.keys():
#     download_file(filename)

aiSideBar = st.sidebar.selectbox(
"Model Selection",
("My Model", "SkyTNT's Model", "TADNE"))

if aiSideBar == "TADNE":
    if "randomState" not in st.session_state: 
        st.session_state.randomState = datetime.datetime.now()
    if "numberState" not in st.session_state: 
        st.session_state.numberState = datetime.datetime.now()
    if "rs_3" not in st.session_state:
        st.session_state.rs_3 = datetime.datetime.now()
    if "Random_State" not in st.session_state:
        st.session_state.Random_State = 65536


    @st.experimental_singleton
    def onnxSessionCreate(g_mapping_path,g_synthesis_path):
        sess_options = rt.SessionOptions()
        sess_options.graph_optimization_level = rt.GraphOptimizationLevel.ORT_ENABLE_ALL
        g_mapping_sess = rt.InferenceSession(g_mapping_path ,sess_options=sess_options,providers=['CPUExecutionProvider',"CUDAExecutionProvider"])
        g_synthesis_sess = rt.InferenceSession(g_synthesis_path ,sess_options=sess_options,providers=['CPUExecutionProvider',"CUDAExecutionProvider"])
        return g_mapping_sess, g_synthesis_sess

    def onnxInference(g_mapping_sess, g_synthesis_sess, truncation, random_state):
        z = np.random.RandomState(random_state).randn(1, 1024).astype("float32")
        w_avg = g_mapping_sess.run(None, {g_mapping_sess.get_inputs()[0].name: z, g_mapping_sess.get_inputs()[1].name: np.array([truncation]).astype(np.float32)})[0]
        img = g_synthesis_sess.run(None, {'latent_w': w_avg})[0]
        return img

    def create_im_num(tensor_arr,files = None):
        img = (tensor_arr + 1) * 255 / 2  # [-1.0, 1.0] -> [0.0, 255.0]
        img = np.transpose(img, [0, 2, 3, 1])
        img = np.clip(img, 0 ,255).astype(np.uint8)[0]
        if files != None:
            PIL.Image.fromarray(img, 'RGB').save(files)
        return PIL.Image.fromarray(img, 'RGB')

    def TADNEdit():
        import pickle
        import numpy as np
        deepdanbooru_dirs = pickle.load(open(r"./models/TADNE/deepdanbooru_dirs.pkl", 'rb'))
        named_directions = {}
        latent_dirs = []

        comp = 0
        for tag, vec in deepdanbooru_dirs.items():
            #print(tag)
            for layers in [(0, 6), (6, 12), (12, 16)]:
                #print(layers[0], layers[1])
                name = tag + "-" + str(layers[0]) + "-" + str(layers[1])
                named_directions[f'{name}'] = [comp, int(layers[0]), int(layers[1]), f'{name}']
            latent_dirs.append(np.matrix(vec[0]))
            comp = comp + 1
        return named_directions, latent_dirs

    def zTADNEforward(g_mapping_sess,random_state,truncation=1.0):
        z = np.random.RandomState(random_state).randn(1, 1024).astype("float32")
        w_avg = g_mapping_sess.run(None, {g_mapping_sess.get_inputs()[0].name: z, g_mapping_sess.get_inputs()[1].name: np.array([truncation]).astype(np.float32)})[0]
        return w_avg

    def wTADNEforward(g_synthesis_sess,w_avg):
        img = g_synthesis_sess.run(None, {'latent_w': w_avg})[0]
        return img

    def updateRandomState(x):
        st.session_state.randomState = datetime.datetime.now()
        st.session_state['number_value'] = st.session_state['truncation_slider'] 
        
    def updateNumberState(x):
        st.session_state.numberState = datetime.datetime.now()
        st.session_state['truncation_slider'] = st.session_state['number_value']

    def updateLastState(x):
        st.session_state.rs3 = datetime.datetime.now()
        try:
            st.session_state['number_value'] = st.session_state['multi_state'][0]
            st.session_state['truncation_slider'] = st.session_state['multi_state'][0]
            st.session_state.Random_State = st.session_state['multi_state'][0]
        except:
            pass


    TRUNC = st.slider("truncation",0.0,2.0,1.0,0.1)
    Random_State1 = st.slider("Random Seed",0,100000,65536,1,on_change=updateRandomState, args=(8,), key='truncation_slider') #args=(something,) for 1 variable
    Random_State2 = st.number_input("Number value", min_value=0, max_value=100000, value=65536, step=1,on_change=updateNumberState, args=(9,), key="number_value")
    options = [26225,62456,63321,97819,82560,36744]
    rs_3 = st.multiselect("Random Seed Choices", options, default=None, key="multi_state", on_change=updateLastState, args=(9,))

    sec = st.session_state.randomState.second - st.session_state.numberState.second
    sec1 = st.session_state.rs_3.second - st.session_state.randomState.second
    sec2 = st.session_state.rs_3.second - st.session_state.numberState.second

    st.write(sec,sec1,sec2)
    Random_State = st.session_state.Random_State
    if sec1 > 0 and sec2 > 0:
        Random_State = rs_3[0]
    elif sec > 0 and sec1 < 0:
        Random_State = Random_State1
    elif sec < 0:
        Random_State = Random_State2
    else:
        pass





    named_directions, latent_dirs = TADNEdit()
    editExpander= st.expander("Editing TADNE")
    with editExpander:
        col1, col2, col3 = st.columns(3)
        with col1:
            for keys,value in named_directions.items():
                if "0-6" in keys:
                    st.slider(f"{keys}",min_value=-10.0,max_value=10.0,value=0.0, step=0.1, key=keys)
        with col2:
            for keys,value in named_directions.items():
                if "6-12" in keys:
                    st.slider(f"{keys}",min_value=-10.0,max_value=10.0,value=0.0, step=0.1, key=keys)

        with col3:
            for keys,value in named_directions.items():
                if "12-16" in keys:
                    st.slider(f"{keys}",min_value=-10.0,max_value=10.0,value=0.0, step=0.1, key=keys)

        ##############################################################################################################
        #TADNE EDITING FUNCTIONS
        def normalize(v):
            norm=np.linalg.norm(v, ord=2)
            if norm==0:
                norm=np.finfo(v.dtype).eps
            return v/norm * len(v)


        def edits(all_W,truncation):
            latent_avg = np.zeros(1024,)
            scale = 1.0
            for tag, value in named_directions.items():
                value = st.session_state[tag]
                
                start_l = named_directions[tag][1]
                end_l = min(16, named_directions[tag][2])
                direction_l = normalize(latent_dirs[named_directions[tag][0]])

                for l in range(start_l, end_l):
                    all_W[0][l] = all_W[0][l] + direction_l * value * scale

            if truncation != 1:
                w_avg = latent_avg
                all_W = w_avg + (all_W - w_avg) * truncation # [minibatch, layer, component]
            
            return all_W
    ################################################################################################################################


    def resetOnClick(x):
        for keys, _ in named_directions.items():
            if st.session_state[keys] != 0.0:
                st.session_state[keys] = 0.0

    def randomiseValues(x):
        from random import randint
        for keys, _ in named_directions.items():
            if st.session_state[keys] == 0.0:
                st.session_state[keys] = float(randint(0, 2))


    # st.write(len(st.session_state))
    # st.write(st.session_state)
    # #st.write(sec)

    modelPath = "./models/TADNE/modelmaptrunc_sim.onnx"
    modelPath2 = "./models/TADNE/modelsynth_sim.onnx"
    g_map, g_syn = onnxSessionCreate(modelPath,modelPath2)

    but = None
    resetButton = None
    randomButton = None

    if editExpander:
        randomButton = st.button("Randomise All Edited Values", on_click=randomiseValues, args=(9,))
        for keys,value in named_directions.items():
            if st.session_state[keys] != 0.0:
                but = st.button("Generate Edited Image", key="editGenerateImage")
                resetButton = st.button("Reset All Edited Values", on_click=resetOnClick, args=(9,))
                break
                
    if but:
        w_avg = zTADNEforward(g_map,Random_State)

        all_W = edits(w_avg,truncation=TRUNC)

        img = create_im_num(wTADNEforward(g_syn,all_W))
        st.image(img)

    if resetButton:
        if resetButton:
            for keys,value in named_directions.items():
                if st.session_state[keys] != 0.0:
                    st.session_state[keys] = 0.0




    button = st.button('Generate Image', key="generateImage")
    if button:
        st.session_state.count = 1
        st.session_state.randomState = datetime.datetime.now()
        st.session_state.numberState = datetime.datetime.now()
        st.session_state.rs_3 = datetime.datetime.now()
        # st.write(st.session_state)
        # st.write(Random_State)
        # st.write(TRUNC)
        img = create_im_num(onnxInference(g_map,g_syn,TRUNC,Random_State))

    # def create_im(tensor_arr,files = None):
    #     img = (tensor_arr + 1) * 255 / 2  # [-1.0, 1.0] -> [0.0, 255.0]
    #     img = img.permute(0, 2, 3, 1).clamp(0, 255).to(torch.uint8).cpu().numpy()[0]  # NCWH => NWHC
    #     if files != None:
    #         PIL.Image.fromarray(img, 'RGB').save(files)
    #     return PIL.Image.fromarray(img, 'RGB')

        st.image(img)















#     # Read in models from the data files.
#     tl_gan_model, feature_names = load_tl_gan_model()
#     session, pg_gan_model = load_pg_gan_model()

#     st.sidebar.title("Features")
#     seed = 27834096
#     # If the user doesn't want to select which features to control, these will be used.
#     default_control_features = ["Young", "Smiling", "Male"]

#     if st.sidebar.checkbox("Show advanced options"):
#         # Randomly initialize feature values.
#         features = get_random_features(feature_names, seed)

#         # Some features are badly calibrated and biased. Removing them
#         block_list = ["Attractive", "Big_Lips", "Big_Nose", "Pale_Skin"]
#         sanitized_features = [
#             feature for feature in features if feature not in block_list
#         ]

#         # Let the user pick which features to control with sliders.
#         control_features = st.sidebar.multiselect(
#             "Control which features?",
#             sorted(sanitized_features),
#             default_control_features,
#         )
#     else:
#         features = get_random_features(feature_names, seed)
#         # Don't let the user pick feature values to control.
#         control_features = default_control_features

#     # Insert user-controlled values from sliders into the feature vector.
#     for feature in control_features:
#         features[feature] = st.sidebar.slider(feature, 0, 100, 50, 5)

#     st.sidebar.title("Note")
#     st.sidebar.write(
#         """Playing with the sliders, you _will_ find **biases** that exist in this
#         model.
#         """
#     )
#     st.sidebar.write(
#         """For example, moving the `Smiling` slider can turn a face from masculine to
#         feminine or from lighter skin to darker. 
#         """
#     )
#     st.sidebar.write(
#         """Apps like these that allow you to visually inspect model inputs help you
#         find these biases so you can address them in your model _before_ it's put into
#         production.
#         """
#     )
#     st.sidebar.caption(f"Streamlit version `{st.__version__}`")

#     # Generate a new image from this feature vector (or retrieve it from the cache).
#     with session.as_default():
#         image_out = generate_image(
#             session, pg_gan_model, tl_gan_model, features, feature_names
#         )

#     st.image(image_out, use_column_width=True)


# def download_file(file_path):
#     # Don't download the file twice. (If possible, verify the download using the file length.)
#     if os.path.exists(file_path):
#         if "size" not in EXTERNAL_DEPENDENCIES[file_path]:
#             return
#         elif os.path.getsize(file_path) == EXTERNAL_DEPENDENCIES[file_path]["size"]:
#             return

#     # These are handles to two visual elements to animate.
#     weights_warning, progress_bar = None, None
#     try:
#         weights_warning = st.warning("Downloading %s..." % file_path)
#         progress_bar = st.progress(0)
#         with open(file_path, "wb") as output_file:
#             with urllib.request.urlopen(
#                 EXTERNAL_DEPENDENCIES[file_path]["url"]
#             ) as response:
#                 length = int(response.info()["Content-Length"])
#                 counter = 0.0
#                 MEGABYTES = 2.0 ** 20.0
#                 while True:
#                     data = response.read(8192)
#                     if not data:
#                         break
#                     counter += len(data)
#                     output_file.write(data)

#                     # We perform animation by overwriting the elements.
#                     weights_warning.warning(
#                         "Downloading %s... (%6.2f/%6.2f MB)"
#                         % (file_path, counter / MEGABYTES, length / MEGABYTES)
#                     )
#                     progress_bar.progress(min(counter / length, 1.0))

#     # Finally, we remove these visual elements by calling .empty().
#     finally:
#         if weights_warning is not None:
#             weights_warning.empty()
#         if progress_bar is not None:
#             progress_bar.empty()


# # Ensure that load_pg_gan_model is called only once, when the app first loads.
# @st.experimental_singleton()
# def load_pg_gan_model():
#     """
#     Create the tensorflow session.
#     """
#     # Open a new TensorFlow session.
#     config = tf.ConfigProto(allow_soft_placement=True)
#     session = tf.Session(config=config)

#     # Must have a default TensorFlow session established in order to initialize the GAN.
#     with session.as_default():
#         # Read in either the GPU or the CPU version of the GAN
#         with open(MODEL_FILE_GPU if USE_GPU else MODEL_FILE_CPU, "rb") as f:
#             G = pickle.load(f)
#     return session, G


# # Ensure that load_tl_gan_model is called only once, when the app first loads.
# @st.experimental_singleton()
# def load_tl_gan_model():
#     """
#     Load the linear model (matrix) which maps the feature space
#     to the GAN's latent space.
#     """
#     with open(FEATURE_DIRECTION_FILE, "rb") as f:
#         feature_direction_name = pickle.load(f)

#     # Pick apart the feature_direction_name data structure.
#     feature_direction = feature_direction_name["direction"]
#     feature_names = feature_direction_name["name"]
#     num_feature = feature_direction.shape[1]
#     feature_lock_status = np.zeros(num_feature).astype("bool")

#     # Rearrange feature directions using Shaobo's library function.
#     feature_direction_disentangled = feature_axis.disentangle_feature_axis_by_idx(
#         feature_direction, idx_base=np.flatnonzero(feature_lock_status)
#     )
#     return feature_direction_disentangled, feature_names


# def get_random_features(feature_names, seed):
#     """
#     Return a random dictionary from feature names to feature
#     values within the range [40,60] (out of [0,100]).
#     """
#     np.random.seed(seed)
#     features = dict((name, 40 + np.random.randint(0, 21)) for name in feature_names)
#     return features


# # Hash the TensorFlow session, the pg-GAN model, and the TL-GAN model by id
# # to avoid expensive or illegal computations.
# @st.experimental_memo(show_spinner=False, ttl=24*60*60)
# def generate_image(_session, _pg_gan_model, _tl_gan_model, features, feature_names):
#     """
#     Converts a feature vector into an image.
#     """
#     # Create rescaled feature vector.
#     feature_values = np.array([features[name] for name in feature_names])
#     feature_values = (feature_values - 50) / 250
#     # Multiply by Shaobo's matrix to get the latent variables.
#     latents = np.dot(_tl_gan_model, feature_values)
#     latents = latents.reshape(1, -1)
#     dummies = np.zeros([1] + _pg_gan_model.input_shapes[1][1:])
#     # Feed the latent vector to the GAN in TensorFlow.
#     with _session.as_default():
#         images = _pg_gan_model.run(latents, dummies)
#     # Rescale and reorient the GAN's output to make an image.
#     images = np.clip(np.rint((images + 1.0) / 2.0 * 255.0), 0.0, 255.0).astype(
#         np.uint8
#     )  # [-1,1] => [0,255]
#     if USE_GPU:
#         images = images.transpose(0, 2, 3, 1)  # NCHW => NHWC
#     return images[0]


# USE_GPU = False
# FEATURE_DIRECTION_FILE = "feature_direction_2018102_044444.pkl"
# MODEL_FILE_GPU = "karras2018iclr-celebahq-1024x1024-condensed.pkl"
# MODEL_FILE_CPU = "karras2018iclr-celebahq-1024x1024-condensed-cpu.pkl"
# EXTERNAL_DEPENDENCIES = {
#     "feature_direction_2018102_044444.pkl": {
#         "url": "https://streamlit-demo-data.s3-us-west-2.amazonaws.com/facegan/feature_direction_20181002_044444.pkl",
#         "size": 164742,
#     },
#     "karras2018iclr-celebahq-1024x1024-condensed.pkl": {
#         "url": "https://streamlit-demo-data.s3-us-west-2.amazonaws.com/facegan/karras2018iclr-celebahq-1024x1024-condensed.pkl",
#         "size": 92338293,
#     },
#     "karras2018iclr-celebahq-1024x1024-condensed-cpu.pkl": {
#         "url": "https://streamlit-demo-data.s3-us-west-2.amazonaws.com/facegan/karras2018iclr-celebahq-1024x1024-condensed-cpu.pkl",
#         "size": 92340233,
#     },
# }

# if __name__ == "__main__":
#     main()
