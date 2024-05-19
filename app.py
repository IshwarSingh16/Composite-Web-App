from tensorflow.keras import Input, layers
from tensorflow.keras.models import Model
from tensorflow.keras import initializers
import keras
from keras.src.models import Sequential
import math
import tensorflow as tf
import numpy as np
import sklearn
import json
from tensorflow.python.ops.variables import trainable_variables
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, Reshape
from tensorflow.keras.regularizers import l2
from sklearn.model_selection import train_test_split
from tensorflow.keras import regularizers
from flask import Flask, render_template
from flask import request
from uuid import uuid4
import os
# from werkzeug.datastructures import V


final_x = np.ones((10, 8, 8, 1))
final_y = np.ones((10, 3))

split_ratio = 0.05
train_images, test_images, train_targets, test_targets = train_test_split(
    final_x, final_y, test_size=split_ratio)

droprate = 0.25

model = Sequential()
model.add(Conv2D(128, (3, 3), padding="same",
          input_shape=train_images[0].shape, activation="relu", name='base_conv_1'))

model.add(Conv2D(128, (3, 3), padding="same",
          activation="relu", name='base_conv_2'))
model.add(Conv2D(64, (3, 3), padding="same",
          activation="relu", name='base_conv_3'))

model.add(Conv2D(64, (3, 3), padding="same",
          activation="relu", name='base_conv_4'))

model.add(Conv2D(64, (3, 3), padding="same",
          activation="relu", name='base_conv_5'))
model.add(Conv2D(64, (3, 3), padding="same",
          activation="relu", name='base_conv_6'))

model.add(Flatten())
model.add(Dense(256, activation="relu",
          kernel_regularizer=regularizers.l2(0), name='base_dense_1'))
model.add(Dense(256, activation="relu",
          kernel_regularizer=regularizers.l2(0), name='base_dense_2'))

model.add(Dense(128, activation='relu',
          kernel_regularizer=regularizers.l2(0), name='base_dense_3'))

model.add(Dense(128, activation='relu',
          kernel_regularizer=regularizers.l2(0), name='base_dense_4'))

model.add(Dense(3, name='base_dense_5'))

model.compile(optimizer='adam',
              loss=tf.keras.losses.mean_squared_error, metrics=['mape'])

checkpoint_path = './grey_checkpoint/grey_point'
model.load_weights(checkpoint_path)

layer_cov_11 = model.get_layer(name='base_conv_1')
layer_cov_11.trainable = False

layer_cov_22 = model.get_layer(name='base_conv_2')
layer_cov_22.trainable = False

layer_cov_33 = model.get_layer(name='base_conv_3')
layer_cov_33.trainable = False

layer_cov_44 = model.get_layer(name='base_conv_4')
layer_cov_44.trainable = False

layer_cov_55 = model.get_layer(name='base_conv_5')
layer_cov_55.trainable = False

layer_cov_66 = model.get_layer(name='base_conv_6')
layer_cov_66.trainable = False

layer_dense_11 = model.get_layer(name='base_dense_1')
layer_dense_11.trainable = False

layer_dense_22 = model.get_layer(name='base_dense_2')
layer_dense_22.trainable = False

layer_dense_33 = model.get_layer(name='base_dense_3')
layer_dense_33.trainable = False

layer_dense_44 = model.get_layer(name='base_dense_4')
layer_dense_44.trainable = False

layer_dense_55 = model.get_layer(name='base_dense_5')
layer_dense_55.trainable = False


app = Flask(__name__)


@app.route("/")
def upload_predict():
    return render_template('html_landing_project_page.html')


@app.route("/composites")
def composites():
    return render_template('Node.html')


@app.route('/proceed', methods=['GET', 'POST'])
def proceed():
    if request.method == 'POST':
        data0 = (request.form.getlist('Array0[]'))
        for i in range(0, len(data0)):
            data0[i] = int(data0[i])
        data1 = (request.form.getlist('Array1[]'))
        for i in range(0, len(data1)):
            data1[i] = int(data1[i])
        data2 = (request.form.getlist('Array2[]'))
        for i in range(0, len(data2)):
            data2[i] = int(data2[i])
        data3 = (request.form.getlist('Array3[]'))
        for i in range(0, len(data3)):
            data3[i] = int(data3[i])
        data4 = (request.form.getlist('Array4[]'))
        for i in range(0, len(data4)):
            data4[i] = int(data4[i])
        data5 = (request.form.getlist('Array5[]'))
        for i in range(0, len(data5)):
            data5[i] = int(data5[i])
        data6 = (request.form.getlist('Array6[]'))
        for i in range(0, len(data6)):
            data6[i] = int(data6[i])
        data7 = (request.form.getlist('Array7[]'))
        for i in range(0, len(data7)):
            data7[i] = int(data7[i])

        data = [data0, data1, data2, data3, data4, data5, data6, data7]

        data = np.array(data)
        data = data.reshape(1, 8, 8, 1)

        # print(data)

        # check_test = np.ones((1, 8, 8, 1))*1
        prediction = model.predict(data)
        # prediction = np.array(prediction)
        # prediction = prediction.reshape(3)
        E11 = prediction[0][0]
        E22 = prediction[0][1]
        G12 = prediction[0][2]

        return 'The calculated mechanical properties of the given composite are:   <br/>  E11: {:.3f}  &nbsp;   E22: {:.3f}   &nbsp;     G12: {:.3f}'.format(E11, E22, G12)

    return render_template('Node.html')


# trial
# global z_noise, image_true, z_vol, input_shape, noise_input, v_max, w_opt, image_output, sum_nodes, q, prop_output, concatenated_layer, model_gen_2
def generate_id():
    return uuid4().__str__()


global Id
# Id = generate_id()
# print(Id)
Id = os.getpid()

# global check, continuetraincheck
# global checkback, continuetraincheckback

# check = continuetraincheck = generate_id()
# # print(check)
# checkback = continuetraincheckback = generate_id()
# # print(checkback)


@app.route("/generative_model_trail", methods=['GET', 'POST'])
def generative_model_trial():
    if request.method == 'POST':

        global n_fib, E1, E2

        step = int(request.form['step'])
        continue_training = int(request.form['continue_train'])
        dropdown_function = int(request.form['dropdown_function'])
        checkId = int(request.form['Identity'])

        # global iscontinued, n_fib_cont, E1_cont, E2_cont, continuetraincheckback, continuetraincheck,
        if (step == 1):
            n_fib = float(request.form['fibres'])
            E1 = float(request.form['E_x'])
            E2 = float(request.form['E_y'])
        else:
            if (Id != checkId):
                return 'error'

        if (continue_training == 1):
            if(Id != checkId):
                return 'error'

        global z_noise, image_true, z_vol, input_shape, noise_input, v_max, w_opt, image_output, sum_nodes, q, prop_output, concatenated_layer, model_gen_2
        # global checkback, check
        # isfirstcall = int(request.form['isfirstcall'])

        # if (step == 1):
        #     checkback = check
        #     # continuetraincheckback = continuetraincheck
        #     # n_fib_cont = n_fib
        #     # E1_cont = E1
        #     # E2_cont = E2
        # else:
        #     if (checkback != check or n_fib != float(request.form['fibres']) or E1 != float(request.form['E_x']) or E2 != float(request.form['E_y'])):
        #         return 'error'

        # if (step == 15):
        #     check = generate_id()
        #     checkback = generate_id()
        # iscontinued = 1

        def main_loss(y_true, y_pred):

            k1 = 1.  # E1_multiplier = k1
            k2 = 1.55  # E2_multiplier = k2

            # user input property
            req_prop = tf.constant([E1, E2, 0.09146526])

            prd_prop = y_pred[0][64:67]

            prop_error_1 = tf.math.reduce_sum(tf.math.square(
                tf.subtract(req_prop[0:2], prd_prop[0:2])))

            w_prop = 1.0
            # loss function for property constraint
            prop_val_loss = tf.multiply(w_prop, prop_error_1)

            # loss functions for property optimization
            prop_error_ratio_loss = tf.math.reduce_sum(
                tf.math.square(tf.subtract(k1*prd_prop[0], k2*prd_prop[1])))
            prop_error_diff = tf.math.reduce_sum(
                tf.subtract(prd_prop[0], prd_prop[1]))
            prop_error_ratio_optimizer = tf.math.reduce_sum(
                tf.divide(prd_prop[1], prd_prop[0]))
            prop_max = -prd_prop[1]

            return prop_val_loss

        def loss_Function(y_true, y_pred):

            # binary loss
            t1 = tf.subtract(1., y_pred)
            t2 = tf.subtract(0., y_pred)
            t3 = y_pred
            mu2 = tf.math.multiply(t1, t3)
            mul = tf.math.multiply(t1, t2)
            sq = tf.math.pow(mu2, 2)
            if (dropdown_function == 1):
                sq = tf.math.pow(mu2, 0.5)

            # volume fraction constraint loss
            rve_sum = tf.math.reduce_sum(y_pred[0][0:64])

            # user input fibre number (can also be input as volume fraction)
            # n_fib
            k = tf.subtract(rve_sum, n_fib)
            vol_constraint = tf.math.square(k)

            # volume fraction optimizer loss
            s1 = -tf.math.reduce_sum(tf.math.square(y_pred[0][0:64]))
            s = tf.math.reduce_sum(tf.math.square(y_pred[0][0:64]))
            s_root = tf.math.reduce_sum(tf.math.sqrt(y_pred[0][0:64]))
            s2 = tf.math.reduce_sum(y_pred[0][0:64])
            s3 = tf.math.pow(s2, 4)

            # weights for each loss functions
            w_vol = 0.1
            w_binary = 10.0
            if (dropdown_function == 1):
                w_binary = 0.05

            vol_loss = tf.multiply(w_vol, vol_constraint)
            binary_loss = tf.multiply(w_binary, sq)
            min_vol_loss = tf.multiply(w_vol, s_root)

            # net loss
            total_constr_loss = tf.add(vol_loss, binary_loss)
            total_opt_loss = tf.add(binary_loss, min_vol_loss)

            if (dropdown_function == 1):
                return binary_loss
            if (dropdown_function == 3):
                return total_opt_loss

            return total_constr_loss

        if (step == 1 and continue_training == 0):
            z_noise = np.ones((1, 100))

            image_true = np.random.rand(1, 64)

            z_vol = np.random.rand(67)
            z_vol = np.reshape(z_vol, (1, 67))

            input_shape = (100,)

            noise_input = Input(shape=input_shape, name='input_layer')

            # to initialize weights
            v_max = 0.9
            w_opt = -(math.log((1-v_max)/v_max))/100.0
            image_output = Dense(64, name='gen_dense_5', activation='sigmoid',
                                 kernel_initializer=initializers.RandomNormal(mean=w_opt, stddev=0.001))(noise_input)
            sum_nodes = Reshape((1, 64))(image_output)

            q = Reshape((8, 8, 1))(image_output)

            q = layer_cov_11(q)
            q = layer_cov_22(q)
            q = layer_cov_33(q)
            q = layer_cov_44(q)
            q = layer_cov_55(q)
            q = layer_cov_66(q)

            q = Flatten()(q)

            q = layer_dense_11(q)
            q = layer_dense_22(q)
            q = layer_dense_33(q)

            q = layer_dense_44(q)
            prop_output = layer_dense_55(q)

            concatenated_layer = tf.keras.layers.concatenate(
                [image_output, prop_output], axis=1)

            model_gen_2 = Model(inputs=[noise_input], outputs=[
                concatenated_layer, image_output])

        # if (step == 1 and continue_training == 0):
        #     checkpoint_path = './Inverse_trained_model/weights'
        #     model_gen_2.load_weights(
        #         checkpoint_path, by_name=False, skip_mismatch=False, options=None)
        # else:
        #     checkpoint_path = './training_model/weights'
        #     model_gen_2.load_weights(
        #         checkpoint_path, by_name=False, skip_mismatch=False, options=None)

        weights = [1.0, 0.1]

        if (dropdown_function == 1):
            weights = [1.0, 1.0]

        if (step == 1):
            model_gen_2.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss=[
                main_loss, loss_Function], metrics=['mse'], loss_weights=weights)

        # model_gen_2.save('saved_model/my_model',
        #                  custom_objects={'loss': (main_loss, loss_Function)})

        # checkpoint_path = 'Inverse_trained_model\cp.ckpt'
        # model_gen_2.load_weights(checkpoint_path)

        # checkpoint_best_path = 'Inverse_trained_model/cp.ckpt'
        # checkpoint_best = ModelCheckpoint(filepath=checkpoint_best_path,
        #                                   save_weights_only=True,
        #                                   verbose=1,
        #                                   )

        # , callbacks=[checkpoint_best]

        history = model_gen_2.fit(
            [z_noise], [z_vol, image_true], epochs=50)

        # filepath = './training_model/weights'
        # model_gen_2.save_weights(
        #     filepath, overwrite=True, save_format=None, options=None)

        a, g = model_gen_2.predict([z_noise])

        output_prop = a[0, 64:67]

        a = np.reshape(a[0][0:64], (8, 8))
        a_1 = np.abs(1.0-a)

        bb = np.copy(a)

        bb = np.reshape(bb, (1, 64))

        ct = 0
        for i in range(0, 64):
            ct = ct+bb[0][i]

        b = a.tolist()

        array_pass = json.dumps(b)

        id_pass = json.dumps(Id)

        if (dropdown_function == 1 or dropdown_function == 3):
            mix = 'Training: &nbsp; Step - {}/15 <br/> Given Properties :  E11: {:.3f}  &nbsp;   E22: {:.3f}   <br/>  Output Properties:   E11: {:.3f}  &nbsp;    E22: {:.3f} &nbsp; No of fibers: {:.3f}  '.format(
                step, E1, E2, output_prop[0], output_prop[1], ct)

        else:
            mix = 'Training: &nbsp; Step - {}/15 <br/> Given Properties :  E11: {:.3f}  &nbsp;   E22: {:.3f} &nbsp; No of fibers: {:.3f}  <br/>  Output Properties:   E11: {:.3f}  &nbsp;    E22: {:.3f} &nbsp; No of fibers: {:.3f}  '.format(
                step, E1, E2,  n_fib, output_prop[0], output_prop[1], ct)

        if (step == 15):
            if (dropdown_function == 1 or dropdown_function == 3):
                mix = 'Training: &nbsp; Completed <br/> Given Properties :  E11: {:.3f}  &nbsp;   E22: {:.3f}   <br/>  Output Properties:   E11: {:.3f}  &nbsp;    E22: {:.3f} &nbsp; No of fibers: {:.3f}  '.format(
                    E1, E2, output_prop[0], output_prop[1], ct)
            else:
                mix = 'Training: &nbsp; Completed <br/> Given Properties :  E11: {:.3f}  &nbsp;   E22: {:.3f} &nbsp; No of fibers: {:.3f}  <br/>  Output Properties:   E11: {:.3f}  &nbsp;    E22: {:.3f} &nbsp; No of fibers: {:.3f}  '.format(
                    E1, E2,  n_fib, output_prop[0], output_prop[1], ct)

        passing = {"array": array_pass,   "print": mix, "id": id_pass}
        passing = json.dumps(passing)

        return (passing)

    return render_template('Node.html')

# end trial


# @app.route("/final_step", methods=['GET', 'POST'])
# def final_step():
#     if request.method == 'POST':
#         z_noise = np.ones((1, 100))

#         image_true = np.random.rand(1, 64)

#         z_vol = np.random.rand(67)
#         z_vol = np.reshape(z_vol, (1, 67))

#         input_shape = (100,)

#         noise_input = Input(shape=input_shape, name='input_layer')

#         # to initialize weights
#         v_max = 0.9
#         w_opt = -(math.log((1-v_max)/v_max))/100.0
#         image_output = Dense(64, name='gen_dense_5', activation='sigmoid',
#                              kernel_initializer=initializers.RandomNormal(mean=w_opt, stddev=0.001))(noise_input)
#         sum_nodes = Reshape((1, 64))(image_output)

#         q = Reshape((8, 8, 1))(image_output)

#         q = layer_cov_11(q)
#         q = layer_cov_22(q)
#         q = layer_cov_33(q)
#         q = layer_cov_44(q)
#         q = layer_cov_55(q)
#         q = layer_cov_66(q)

#         q = Flatten()(q)

#         q = layer_dense_11(q)
#         q = layer_dense_22(q)
#         q = layer_dense_33(q)

#         q = layer_dense_44(q)
#         prop_output = layer_dense_55(q)

#         concatenated_layer = tf.keras.layers.concatenate(
#             [image_output, prop_output], axis=1)

#         model_gen_2 = Model(inputs=[noise_input], outputs=[
#             concatenated_layer, image_output])

#         filepath = './Inverse_trained_model/weights'

#         model_gen_2.save_weights(
#             filepath, overwrite=True, save_format=None, options=None)

#         return 'Initialized again'


if __name__ == "__main__":
    app.run(use_reloader=False, debug=True)
    # port=11500, debug=True,
# debug true shows error inside the web browser
