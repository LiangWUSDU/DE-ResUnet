from keras import layers, models,Model
from keras import backend as K
from keras.initializers import RandomNormal,Constant
from keras.layers import Conv3D,Reshape,Add,LeakyReLU,Multiply,Lambda,GlobalAveragePooling3D,Average,Dense,multiply,Maximum,Subtract,UpSampling3D,AveragePooling3D,concatenate,Permute,Input,BatchNormalization,add,Conv3DTranspose,Activation,MaxPooling3D,Dropout
K.set_image_data_format('channels_last')
import tensorflow as tf
def BatchActivate(x):
    x = BatchNormalization()(x)
#    x = Activation('relu')(x)
    x = LeakyReLU(0.2)(x)
    return x

def res_block(x, nb_filters, strides):
	res_path = BatchActivate(x)
	res_path = Conv3D(filters=nb_filters[0], kernel_size=(3, 3, 3), padding='same', strides=strides[0], kernel_initializer='he_normal')(res_path)
	res_path = BatchActivate(res_path)
	res_path = Conv3D(filters=nb_filters[1], kernel_size=(3, 3 ,3), padding='same', strides=strides[1],kernel_initializer='he_normal')(res_path)

	shortcut = Conv3D(nb_filters[1], kernel_size=(1, 1, 1), strides=strides[0],kernel_initializer='he_normal')(x)

	res_path = add([shortcut, res_path])
	return res_path


def encoder(x):
    to_decoder = []

    main_path = Conv3D(filters=16, kernel_size=(3, 3, 3), padding='same', strides=(1, 1, 1),kernel_initializer='he_normal')(x)
    main_path = BatchActivate(main_path)

    main_path = Conv3D(filters=16, kernel_size=(3, 3, 3), padding='same', strides=(1, 1, 1),kernel_initializer='he_normal')(main_path)

    shortcut = Conv3D(filters=16, kernel_size=(1, 1, 1), strides=(1, 1, 1),kernel_initializer='he_normal')(x)

    main_path = add([shortcut, main_path])
     # first branching to decoder
    to_decoder.append(main_path)

    main_path = res_block(main_path, [32, 32], [(2, 2,2), (1,1, 1)])
    to_decoder.append(main_path)

    main_path = res_block(main_path, [64, 64], [(2, 2,2), (1, 1,1)])
    to_decoder.append(main_path)


    main_path = res_block(main_path, [128, 128], [(2, 2, 2), (1, 1, 1)])
    to_decoder.append(main_path)

    return to_decoder


def decoder(x, from_encoder,from_encoderF):
    main_path = UpSampling3D(size=(2, 2, 2))(x)
    main_path = concatenate([main_path, from_encoderF[3],from_encoder[3]], axis=-1)
    main_path = res_block(main_path, [128, 128], [(1, 1,1), (1,1, 1)])

    main_path = UpSampling3D(size=(2, 2, 2))(main_path)
    main_path = concatenate([main_path, from_encoderF[2],from_encoder[2]], axis=-1)
    main_path = res_block(main_path, [64, 64], [(1, 1,1), (1, 1,1)])

    main_path = UpSampling3D(size=(2, 2, 2))(main_path)
    main_path = concatenate([main_path, from_encoderF[1],from_encoder[1]], axis=-1)
    main_path = res_block(main_path, [32, 32], [(1, 1,1), (1, 1,1)])


    main_path = UpSampling3D(size=(2, 2, 2))(main_path)
    main_path = concatenate([main_path, from_encoderF[0], from_encoder[0]], axis=-1)
    main_path = res_block(main_path, [16, 16], [(1, 1, 1), (1, 1, 1)])

    return main_path
def res_block_attention(x, filter):
	res_path = Conv3D(filters=filter, kernel_size=(3, 3, 3), padding='same',kernel_initializer='he_normal')(x)
	res_path = BatchActivate(res_path)
	res_path = Conv3D(filters=filter, kernel_size=(3, 3 ,3), padding='same',activation='relu',kernel_initializer='he_normal')(res_path)
	shortcut = Conv3D(filter, kernel_size=(1, 1, 1))(x)
	shortcut =  LeakyReLU(0.2)(shortcut)
	res_path = add([shortcut, res_path])
	return res_path
def my_attention(X,Y,Z):
    #X某一类组织
    # Y 背景
    x0 = concatenate([X,Z],axis=-1)
    x1 = res_block_attention(x0, 32)
    x2 = res_block_attention(x1,32)
    x3 = Conv3D(1,kernel_size=(1,1,1),padding='same',activation='sigmoid')(x2)
    x4 = Lambda(lambda x: 1+x)(x3)
    x5 = Multiply()([x4,X])
    y1 = Lambda(lambda x: 2-x)(Y)
    y2 =Multiply()([y1,x3])
    z = add([x5,y2])
    z = res_block_attention(z, 32)
    return z

def mutil_scale_decoder(x, from_encoder,from_encoderF):
    main_path1 = UpSampling3D(size=(2, 2, 2))(x)
    main_path1 = concatenate([main_path1, from_encoderF[3],from_encoder[3]], axis=-1)
    main_path1 = se_block(main_path1,512)
    main_path1 = res_block(main_path1, [128, 128], [(1, 1,1), (1,1, 1)])       #8,8,8,128

    main_path2 = UpSampling3D(size=(2, 2, 2))(main_path1)
    main_path2 = concatenate([main_path2, from_encoderF[2],from_encoder[2]], axis=-1)
    main_path2 = se_block(main_path2,256)
    main_path2 = res_block(main_path2, [64, 64], [(1, 1,1), (1, 1,1)]) #16,16,16,64

    main_path3 = UpSampling3D(size=(2, 2, 2))(main_path2)
    main_path3 = concatenate([main_path3, from_encoderF[1],from_encoder[1]], axis=-1)
    main_path3 = se_block(main_path3,128)
    main_path3 = res_block(main_path3, [32, 32], [(1, 1,1), (1, 1,1)])  #32,32,32,32


    main_path4 = UpSampling3D(size=(2, 2, 2))(main_path3)
    main_path4 = concatenate([main_path4, from_encoderF[0], from_encoder[0]], axis=-1)
    main_path4 = se_block(main_path4,64)
    main_path4 = res_block(main_path4, [16, 16], [(1, 1, 1), (1, 1, 1)])  #64,64,64,16

    return main_path1,main_path2,main_path3,main_path4

#def Sharpen(y):
def se_block(x, filters, ratio=16):
    """
    creates a squeeze and excitation block
    https://arxiv.org/abs/1709.01507

    Parameters
    ----------
    x : tensor
        Input keras tensor
    ratio : int
        The reduction ratio. The default is 16.
    Returns
    -------
    x : tensor
        A keras tensor
    """

    se_shape = (1,1,1, filters)

    se = GlobalAveragePooling3D()(x)
    se = Reshape(se_shape)(se)
    se = Dense(filters // ratio, activation='relu', kernel_initializer='he_normal', use_bias=False)(se)
    se = Dense(filters, activation='sigmoid', kernel_initializer='he_normal', use_bias=False)(se)

    x = multiply([x, se])
    return x


def ResNetR3_attention_mutil_scale(input_shape, input_shape1,input_shape2,input_shape3,input_shape4):

    inputs = Input(shape=input_shape)  #T1
    I = Conv3D(16, kernel_size=(3, 3, 3), padding='same', kernel_initializer='he_normal')(inputs)
    I = LeakyReLU(0.2)(I)
    inputs1 = Input(shape=input_shape1)
    I1 = Conv3D(16, kernel_size=(3, 3, 3), padding='same', kernel_initializer='he_normal')(inputs1)
    I1 = LeakyReLU(0.2)(I1)
    inputs2 = Input(shape=input_shape2)
    I2 = Conv3D(16, kernel_size=(3, 3, 3), padding='same', kernel_initializer='he_normal')(inputs2)
    I2 = LeakyReLU(0.2)(I2)
    inputs3 = Input(shape=input_shape3)
    I3 = Conv3D(16, kernel_size=(3, 3, 3), padding='same', kernel_initializer='he_normal')(inputs3)
    I3 = LeakyReLU(0.2)(I3)
    inputs4 = Input(shape=input_shape4)
    I4 = Conv3D(16, kernel_size=(3, 3, 3), padding='same', kernel_initializer='he_normal')(inputs4)
    I4 = LeakyReLU(0.2)(I4)
    inputsF = concatenate([I1,I2,I3,I4],axis=-1)
    to_decoder1 = encoder(I)
    to_decoder2 = encoder(inputsF)
    to_decoder3 = concatenate([to_decoder1[3],to_decoder2[3]],axis=-1)
    path1 = res_block(to_decoder3, [256, 256], [(2, 2, 2), (1, 1, 1)])
    path1, path2, path3, path4 = mutil_scale_decoder(path1, from_encoder=to_decoder1,from_encoderF=to_decoder2)

    x001 = Conv3DTranspose(16,kernel_size=(1,1,1), activation='relu', strides=(8,8,8), kernel_initializer='he_normal',bias_initializer=Constant(value=-10))(path1)
    x001 = res_block_attention(x001,32)
    x8 = Conv3D(4, kernel_size=(1, 1, 1), activation='softmax',name='output8',   kernel_initializer='he_normal',bias_initializer=Constant(value=-10))(x001)

    x002 = Conv3DTranspose(16,kernel_size=(1,1,1), activation='relu', strides=(4,4,4), kernel_initializer='he_normal',bias_initializer=Constant(value=-10))(path2)
    x002 = res_block_attention(x002,32)
    x16 = Conv3D(4, kernel_size=(1, 1, 1), activation='softmax',name='output16', kernel_initializer='he_normal',bias_initializer=Constant(value=-10))(x002)

    x003 = Conv3DTranspose(16,kernel_size=(1,1,1), activation='relu', strides=(2,2,2), kernel_initializer='he_normal',bias_initializer=Constant(value=-10))(path3)
    x003 = res_block_attention(x003,32)
    x32 = Conv3D(4, kernel_size=(1, 1, 1), activation='softmax',name='output32', kernel_initializer='he_normal',bias_initializer=Constant(value=-10))(x003)

    x64 = Conv3D(4, kernel_size=(1, 1, 1), activation='softmax', name='output64',  kernel_initializer='he_normal',bias_initializer=Constant(value=-10))(path4)
    x = concatenate([x8,x16,x32,x64],axis=-1)
    x = res_block_attention(x,64)
    x = Conv3D(4, kernel_size=(1, 1, 1), activation='softmax', name='output1',  kernel_initializer='he_normal',bias_initializer=Constant(value=-10))(x)

    BK = Lambda(lambda x:  x[...,0:1])(x)
    CSF = Lambda(lambda x: x[...,1:2])(x)
    GM = Lambda(lambda x:  x[...,2:3])(x)
    WM = Lambda(lambda x:  x[...,3:4])(x)
    CSF = my_attention(CSF,BK,inputs)
    GM = my_attention(GM,BK,inputs)
    WM = my_attention(WM,BK,inputs)
    combin = concatenate([BK, CSF, GM, WM], axis=-1)
    combin = res_block_attention(combin,32)
    output = Conv3D(4, kernel_size=(1, 1, 1), activation='softmax', name='output2',kernel_initializer='he_normal',bias_initializer=Constant(value=-10))(combin)

    CSF = Lambda(lambda x: x[..., 1:2], name='CSF')(output)
    GM =  Lambda(lambda x: x[..., 2:3], name='GM')(output)
    WM = Lambda(lambda x: x[..., 3:4], name='WM')(output)
    model = Model(inputs=[inputs,inputs1,inputs2,inputs3,inputs4], outputs=[x8,x16,x32,x64,x,output,CSF,GM,WM])
    return model
