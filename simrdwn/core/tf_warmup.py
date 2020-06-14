# #!/usr/bin/env python2
# # -*- coding: utf-8 -*-
# """
# Created on Fr June 12.06.2020

# @author: danielbarco
# """
# This Code has not been tested nor completed and merely includes suggestions !
#
# import tensorflow as tf
# import requests
# import base64

# from tensorflow.python.framework import tensor_util
# from tensorflow_serving.apis import predict_pb2
# from tensorflow_serving.apis import prediction_log_pb2


# def get_image_bytes():
#     image_content = requests.get(IMAGE_URL, stream=True)
#     image_content.raise_for_status()
#     return image_content.content



# def warm_up(path_train_folder_cut, num_img):
#     """Generate TFRecords for warming up."""

#     with tf.io.TFRecordWriter("tf_serving_warmup_requests") as writer:
#         image_bytes = get_image_bytes()
#         predict_request = predict_pb2.PredictRequest()
#         predict_request.model_spec.name = 'resnet'
#         predict_request.model_spec.signature_name = 'serving_default'
#         predict_request.inputs['image_bytes'].CopyFrom(
#             tensor_util.make_tensor_proto([image_bytes], tf.string))        
#         log = prediction_log_pb2.PredictionLog(
#             predict_log=prediction_log_pb2.PredictLog(request=predict_request))
#         for r in range(num_img):
#             writer.write(log.SerializeToString())    


#     with tf.python_io.TFRecordWriter("tf_serving_warmup_requests") as writer:
#         # replace <request> with one of:
#         # predict_pb2.PredictRequest(..)
#         # classification_pb2.ClassificationRequest(..)
#         # regression_pb2.RegressionRequest(..)
#         # inference_pb2.MultiInferenceRequest(..)
#         log = prediction_log_pb2.PredictionLog(
#             predict_log=prediction_log_pb2.PredictLog(request=<request>))
#         writer.write(log.SerializeToString())



 