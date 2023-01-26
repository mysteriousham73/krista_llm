import torch
import time
import re

from transformers import T5Tokenizer, T5ForConditionalGeneration
#
#     Copyright (C) 2019-present Nathan Odle
#
#     This program is free software: you can redistribute it and/or modify
#     it under the terms of the Server Side Public License, version 1,
#     as published by MongoDB, Inc.
#
#     This program is distributed in the hope that it will be useful,
#     but WITHOUT ANY WARRANTY; without even the implied warranty of
#     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#     Server Side Public License for more details.
#
#     You should have received a copy of the Server Side Public License
#     along with this program. If not, email mysteriousham73@gmail.com
#
#     As a special exception, the copyright holders give permission to link the
#     code of portions of this program with the OpenSSL library under certain
#     conditions as described in each individual source file and distribute
#     linked combinations including the program with the OpenSSL library. You
#     must comply with the Server Side Public License in all respects for
#     all of the code used other than as permitted herein. If you modify file(s)
#     with this exception, you may extend this exception to your version of the
#     file(s), but you are not obligated to do so. If you do not wish to do so,
#     delete this exception statement from your version. If you delete this
#     exception statement from all source files in the program, then also delete
#     it in the license file.

from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn

tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-xxl", cache_dir="/media/models/.cache/huggingface/hub")
model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-xxl", cache_dir="/media/models/.cache/huggingface/hub", device_map="auto", load_in_8bit=True)
print(model.get_memory_footprint())
print(model.hf_device_map)


app = FastAPI()

if __name__ == "__main__":

    config = uvicorn.Config("main:app", host="192.168.1.2", port=8001, log_level="info")
    server = uvicorn.Server(config)
    server.run()


class InferenceRequest(BaseModel):
    query: str

class InferenceResponse(BaseModel):
    response: str
    time_to_generate: float

    # "This is a conversation about satellites.  Random conversation Seed: 2242424244.  Conversational answers include a greeting, a verbose answer.  You know the range of PICSAT from your position is 527km.  provide a conversational answer to the question <how far away is PICSAT right now>"
    # "possible parameters: range, latitude, longitude, elevation, azimuth, altitude, frequencies.  you are a chatbot with this request: when will the iss be by again?  use this format for your response:  {satellite_name: <satellite_name>, query_type: <query_type>, request_parameters: <request_parameters>}  query_type should be one of: satellite_pass_prediction, satellite_status. satellite_pass_prediction is when the satellite will be overhead. satellite_status is about the satellite state.  request_parameters is a list of requested information. possible request_parameters are part of the request.  {satellite_name:"
    # "Only azimuth and elevation are relevant to antennas. maximum elevation, rise time, set time are relevant to satellite passes.  \n\n Sat1 information: pass_duration:  how long a pass is  range: distance from user, elevation:  angle above horizon, azimuth: angle from north, altitude: how high above earth's surface, latitude, longitude. maximum elevation:  what is the highest elevation above the horizon for a pass.  rise time:  when does the satellite rise above the horizon for a pass.  set time:  when does the satellite set below the horizon for a pass. Which information parts are relevant to the question <for how long will sat1 be overhead>"
@app.post("/infer/")
async def infer(inference_request: InferenceRequest):

    t_start = time.perf_counter()

    input = inference_request.query.upper()

    #gpt_prompt = "query_type should be one of: satellite_pass_prediction, satellite_status. turn the question <where the heck is horyu-4 right now> into valid JSON containing only satellite_name and query_type. {"
    gpt_prompt = input
    input_ids = tokenizer(gpt_prompt, return_tensors="pt").input_ids.to("cuda")

    gen_tokens = model.generate(input_ids, do_sample=True, temperature=0.0001, max_new_tokens=300,)
    gen_text = tokenizer.batch_decode(gen_tokens)[0]

    t_generate = time.perf_counter()
    response = InferenceResponse(request=input, response=gen_text, time_to_generate=t_generate - t_start)

    return response