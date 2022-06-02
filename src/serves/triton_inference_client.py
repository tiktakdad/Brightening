#!/usr/bin/env python
# Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import cv2
import numpy as np
from PIL import Image
import sys
from functools import partial
import os

import tritonclient.grpc as grpcclient
import tritonclient.grpc.model_config_pb2 as mc
import tritonclient.http as httpclient
from tritonclient.utils import triton_to_np_dtype
from tritonclient.utils import InferenceServerException
from util_yolov5 import letterbox_image, postprocess_yolov5, display_output, output_to_json, output_result

if sys.version_info >= (3, 0):
    import queue
else:
    import Queue as queue


class UserData:

    def __init__(self):
        self._completed_requests = queue.Queue()


# Callback function used for async_stream_infer()
def completion_callback(user_data, result, error):
    # passing error raise and handling out
    user_data._completed_requests.put((result, error))


FLAGS = None


def parse_model_grpc(model_metadata, model_name, model_config):
    """
    Check the configuration of a model to make sure it meets the
    requirements for an image classification network (as expected by
    this client)
    """
    if len(model_config.input) != 1:
        raise Exception("expecting 1 input, got {}".format(len(model_config.input)))

    input_config = model_config.input[0]
    input_metadata = model_metadata.inputs[0]

    for output in model_metadata.outputs:
        if output.datatype != "FP32":
            raise Exception("expecting output datatype to be TYPE_FP32, model '" +
                            model_name + "' output type is " +
                            model_config.DataType.Name(output.data_type))

    output_names = [output.name for output in model_config.output]
    # Model specifying maximum batch size of 0 indicates that batching
    # is not supported and so the input tensors do not expect an "N"
    # dimension (and 'batch_size' should be 1 so that only a single
    # image instance is inferred at a time).

    input_batch_dim = (model_config.max_batch_size > 0)
    expected_input_dims = 3 + (1 if input_batch_dim else 0)
    if len(input_metadata.shape) != expected_input_dims:
        raise Exception(
            "expecting input to have {} dimensions, model '{}' input has {}".
                format(expected_input_dims, model_metadata.name,
                       len(input_metadata.shape)))

    if ((input_config.format != mc.ModelInput.FORMAT_NCHW) and
            (input_config.format != mc.ModelInput.FORMAT_NHWC)):
        raise Exception("unexpected input format " +
                        mc.ModelInput.Format.Name(input_config.format) +
                        ", expecting " +
                        mc.ModelInput.Format.Name(mc.ModelInput.FORMAT_NCHW) +
                        " or " +
                        mc.ModelInput.Format.Name(mc.ModelInput.FORMAT_NHWC))

    if input_config.format == mc.ModelInput.FORMAT_NHWC:
        h = input_metadata.shape[1 if input_batch_dim else 0]
        w = input_metadata.shape[2 if input_batch_dim else 1]
        c = input_metadata.shape[3 if input_batch_dim else 2]
    else:
        c = input_metadata.shape[1 if input_batch_dim else 0]
        h = input_metadata.shape[2 if input_batch_dim else 1]
        w = input_metadata.shape[3 if input_batch_dim else 2]

    return (model_config.max_batch_size, input_config.name, output_names, c, h, w,
            input_config.format, input_metadata.datatype)


def parse_model_http(model_metadata, model_config):
    """
    Check the configuration of a model to make sure it meets the
    requirements for an image classification network (as expected by
    this client)
    """
    if len(model_metadata['inputs']) != 1:
        raise Exception("expecting 1 input, got {}".format(
            len(model_metadata['inputs'])))
    if len(model_metadata['outputs']) != 1:
        raise Exception("expecting 1 output, got {}".format(
            len(model_metadata['outputs'])))

    if len(model_config['input']) != 1:
        raise Exception(
            "expecting 1 input in model configuration, got {}".format(
                len(model_config['input'])))

    input_metadata = model_metadata['inputs'][0]
    input_config = model_config['input'][0]
    output_metadata = model_metadata['outputs'][0]

    max_batch_size = 0
    if 'max_batch_size' in model_config:
        max_batch_size = model_config['max_batch_size']

    if output_metadata['datatype'] != "FP32":
        raise Exception("expecting output datatype to be FP32, model '" +
                        model_metadata['name'] + "' output type is " +
                        output_metadata['datatype'])

    # Output is expected to be a vector. But allow any number of
    # dimensions as long as all but 1 is size 1 (e.g. { 10 }, { 1, 10
    # }, { 10, 1, 1 } are all ok). Ignore the batch dimension if there
    # is one.
    output_batch_dim = (max_batch_size > 0)
    non_one_cnt = 0
    for dim in output_metadata['shape']:
        if output_batch_dim:
            output_batch_dim = False
        elif dim > 1:
            non_one_cnt += 1
            if non_one_cnt > 1:
                raise Exception("expecting model output to be a vector")

    # Model input must have 3 dims (not counting the batch dimension),
    # either CHW or HWC
    input_batch_dim = (max_batch_size > 0)
    expected_input_dims = 3 + (1 if input_batch_dim else 0)
    if len(input_metadata['shape']) != expected_input_dims:
        raise Exception(
            "expecting input to have {} dimensions, model '{}' input has {}".
            format(expected_input_dims, model_metadata['name'],
                   len(input_metadata['shape'])))

    if ((input_config['format'] != "FORMAT_NCHW") and
        (input_config['format'] != "FORMAT_NHWC")):
        raise Exception("unexpected input format " + input_config['format'] +
                        ", expecting FORMAT_NCHW or FORMAT_NHWC")

    if input_config['format'] == "FORMAT_NHWC":
        h = input_metadata['shape'][1 if input_batch_dim else 0]
        w = input_metadata['shape'][2 if input_batch_dim else 1]
        c = input_metadata['shape'][3 if input_batch_dim else 2]
    else:
        c = input_metadata['shape'][1 if input_batch_dim else 0]
        h = input_metadata['shape'][2 if input_batch_dim else 1]
        w = input_metadata['shape'][3 if input_batch_dim else 2]

    return (max_batch_size, input_metadata['name'], output_metadata['name'], c,
            h, w, input_config['format'], input_metadata['datatype'])


def preprocess(img, format, dtype, c, h, w, protocol):
    """
    Pre-process an image to meet the size, type and format
    requirements specified by the parameters.
    """
    # np.set_printoptions(threshold='nan')

    if c == 1:
        sample_img = img.convert('L')
    else:
        sample_img = img.convert('RGB')

    #sample_img = np.array(sample_img)
    #sample_img = sample_img[:, :, ::-1].copy()
    resized = letterbox_image(sample_img, (w, h))

    npdtype = triton_to_np_dtype(dtype)
    img_in =  np.array(resized).astype(npdtype)  # HWC -> CHW
    # img_in = np.transpose(resized, (2, 0, 1)).astype(np.float32)  # HWC -> CHW
    #img_in = np.expand_dims(img_in, axis=0)
    img_in /= 255.0

    if protocol == "grpc":
        if format == mc.ModelInput.FORMAT_NCHW:
            ordered = np.transpose(img_in, (2, 0, 1))
        else:
            ordered = img_in
    else:
        if format == "FORMAT_NCHW":
            ordered = np.transpose(img_in, (2, 0, 1))
        else:
            ordered = img_in




    # Swap to CHW if necessary


    # Channels are in RGB order. Currently model configuration data
    # doesn't provide any information as to other channel orderings
    # (like BGR) so we just assume RGB.
    return ordered



def postprocess(results, output_names, batch_size, c, h, w, batching):
    output_arrays = []
    for output_name in output_names:
        output_array = results.as_numpy(output_name)
        if len(output_array) != batch_size:
            raise Exception("expected {} results, got {}".format(
                batch_size, len(output_array)))
        output_arrays.append(output_array)


    detections = postprocess_yolov5(output_arrays, w, h, batch_size)

    """
    Post-process results to show classifications.
    """
    '''
    output_arrays = []
    for output_name in output_names:
        output_array = results.as_numpy(output_name)
        if len(output_array) != batch_size:
            raise Exception("expected {} results, got {}".format(
                batch_size, len(output_array)))
        output_arrays.append(output_array)

    ltrd_list = get_ltrd_class_acc_from_ext_roi_layer(output_arrays, 0.5, h, w, num_total_layers=4, num_tail_types=2)
    loc_conf = run_fast_nms(ltrd_list, 0.3, 0.0)
    '''
    return detections
    # Include special handling for non-batching models



def requestGenerator(batched_image_data, input_name, output_names, dtype, protocol, model_name, model_version):
    # Set the input data
    inputs = []
    if protocol.lower() == "grpc":
        inputs.append(
            grpcclient.InferInput(input_name, batched_image_data.shape, dtype))
        inputs[0].set_data_from_numpy(batched_image_data)
    else:
        inputs.append(
            httpclient.InferInput(input_name, batched_image_data.shape, dtype))
        inputs[0].set_data_from_numpy(batched_image_data, binary_data=True)

    outputs = []
    for output_name in output_names:
        if protocol.lower() == "grpc":
            outputs.append(
                grpcclient.InferRequestedOutput(output_name))
        else:
            outputs.append(
                httpclient.InferRequestedOutput(output_name,
                                                binary_data=True))

    yield inputs, outputs, model_name, model_version


def run_tis(model_name, url, protocol, image_filenames, model_version='', batch_size=1, streaming=True, async_set=False,
            verbose=False):
    try:
        if protocol == "grpc":
            # Create gRPC client for communicating with the server
            triton_client = grpcclient.InferenceServerClient(
                url=url, verbose=verbose)
        else:
            # Specify large enough concurrency to handle the
            # the number of requests.
            concurrency = 20 if async_set else 1
            triton_client = httpclient.InferenceServerClient(
                url=url, verbose=verbose, concurrency=concurrency)
    except Exception as e:
        print("client creation failed: " + str(e))
        sys.exit(1)

        # Make sure the model matches our requirements, and get some
        # properties of the model that we need for preprocessing
    try:
        model_metadata = triton_client.get_model_metadata(
            model_name=model_name, model_version=model_version)
    except InferenceServerException as e:
        print("failed to retrieve the metadata: " + str(e))
        sys.exit(1)
    try:
        model_config = triton_client.get_model_config(
            model_name=model_name, model_version=model_version)
    except InferenceServerException as e:
        print("failed to retrieve the config: " + str(e))
        sys.exit(1)

    if protocol.lower() == "grpc":
        max_batch_size, input_name, output_names, c, h, w, format, dtype = parse_model_grpc(
            model_metadata, model_name, model_config.config)
    else:
        max_batch_size, input_name, output_name, c, h, w, format, dtype = parse_model_http(
            model_metadata, model_config)

    filenames = []
    if os.path.isdir(image_filenames):
        filenames = [
            os.path.join(image_filenames, f)
            for f in os.listdir(image_filenames)
            if os.path.isfile(os.path.join(image_filenames, f))
        ]
    else:
        filenames = [
            image_filenames,
        ]

    filenames.sort()

    # Preprocess the images into input data according to model
    # requirements
    image_data = []
    imgs = []
    for filename in filenames:
        #img = Image.open(filename)
        #img = static_img
        imgs.append(img)
        image_data.append(preprocess(img, format, dtype, c, h, w, protocol.lower()))
        #image_data.append(np.zeros((3,640,640), dtype=np.float32))

    # Send requests of batch_size images. If the number of
    # images isn't an exact multiple of batch_size then just
    # start over with the first images until the batch is filled.
    requests = []
    responses = []
    result_filenames = []
    request_ids = []
    image_idx = 0
    last_request = False
    user_data = UserData()

    # Holds the handles to the ongoing HTTP async requests.
    async_requests = []

    sent_count = 0

    if streaming:
        triton_client.start_stream(partial(completion_callback, user_data))

    while not last_request:
        input_filenames = []
        repeated_image_data = []

        for idx in range(batch_size):
            input_filenames.append(filenames[image_idx])
            repeated_image_data.append(image_data[image_idx])
            image_idx = (image_idx + 1) % len(image_data)
            if image_idx == 0:
                last_request = True

        if max_batch_size > 0:
            batched_image_data = np.stack(repeated_image_data, axis=0)
        else:
            batched_image_data = repeated_image_data[0]

        # Send request
        try:
            for inputs, outputs, model_name, model_version in requestGenerator(
                    batched_image_data, input_name, output_names, dtype, protocol, model_name, model_version):
                sent_count += 1
                if streaming:
                    triton_client.async_stream_infer(
                        model_name,
                        inputs,
                        request_id=str(sent_count),
                        model_version=model_version,
                        outputs=outputs)
                elif async_set:
                    if protocol.lower() == "grpc":
                        triton_client.async_infer(
                            model_name,
                            inputs,
                            partial(completion_callback, user_data),
                            request_id=str(sent_count),
                            model_version=model_version,
                            outputs=outputs)
                    else:
                        async_requests.append(
                            triton_client.async_infer(
                                model_name,
                                inputs,
                                request_id=str(sent_count),
                                model_version=model_version,
                                outputs=outputs))
                else:
                    responses.append(
                        triton_client.infer(model_name,
                                            inputs,
                                            request_id=str(sent_count),
                                            model_version=model_version,
                                            outputs=outputs))

        except InferenceServerException as e:
            print("inference failed: " + str(e))
            if streaming:
                triton_client.stop_stream()
            sys.exit(1)

    if streaming:
        triton_client.stop_stream()

    if protocol.lower() == "grpc":
        if streaming or async_set:
            processed_count = 0
            while processed_count < sent_count:
                (results, error) = user_data._completed_requests.get()
                processed_count += 1
                if error is not None:
                    print("inference failed: " + str(error))
                    sys.exit(1)
                responses.append(results)
    else:
        if async_set:
            # Collect results from the ongoing async requests
            # for HTTP Async requests.
            for async_request in async_requests:
                responses.append(async_request.get_result())

    for response in responses:
        if protocol.lower() == "grpc":
            this_id = response.get_response().id
        else:
            this_id = response.get_response()["id"]
        # print("Request {}, batch size {}".format(this_id, batch_size))

        detections = postprocess(response, output_names, batch_size, c, h, w, max_batch_size > 0)
        #result = output_result(detections[0], 640, 640)
        #json_result = output_to_json(result)
        return 'pass'

