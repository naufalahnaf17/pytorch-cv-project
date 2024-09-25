'use client'

import { useState,useEffect } from "react";
import {resizeImage} from '../helper/index'
import * as ort from 'onnxruntime-web';

export default function GridImage() {
    const [image,setImage] = useState(null)
    const [session,setSession] = useState(null)
    const [loading,setLoading] = useState(true)
    const [prob,setProb] = useState(null)

    useEffect(() => {
        setSessionModel()
    },[])

    const setSessionModel = async () => {
        const session = await ort.InferenceSession.create('../cat-and-dogs.onnx')
        setSession(session)
        setLoading(false)
    }

    const addImage = (e) => {
        setProb(null)
        setImage(null)

        let imageRaw = e.target.files[0]
        let reader = new FileReader()
        reader.onload = (x) => {
            setImage(x.target.result)

            let base64image = x.target.result.split(",")[1]
            preprocessImage(base64image)
        }
        reader.readAsDataURL(imageRaw)
    }

    const preprocessImage = async (image) => {
        // resize image with sharp
        let uint8image = await resizeImage(image)

        // transponse image to rgb
        const [red,green,blue] = new Array(new Array(), new Array(), new Array())
        for (let i = 0 ; i < uint8image.length ; i+=3){
            red.push(uint8image[i])
            green.push(uint8image[i + 1])
            blue.push(uint8image[i + 2])
        }

        const transposeImage = red.concat(green).concat(blue)

        // turn image to tensor and normalize
        let float32image = new Float32Array(uint8image.length)
        for (let i = 0 ; i < transposeImage.length ; i++){
            float32image[i] = transposeImage[i] / 255.0
        }

        const imageTensor = await new ort.Tensor('float32',float32image,[1,3,224,224])
        inferenceImage(imageTensor)
    }

    const softmax = (arr) => {
        const expScores = arr.map(Math.exp);
        const sumExpScores = expScores.reduce((a, b) => a + b, 0);
        return expScores.map(score => score / sumExpScores);
    };

    const inferenceImage = async (tensor) => {
        const pred = await session.run({ input : tensor })
        const prob = softmax(pred.output.data)
        setProb(prob)
    }

    if (loading){
        return(
            <div className="w-full h-full flex items-center justify-center">
                <p className="text-3xl font-mono">Loading Models</p>
                <svg className="animate-spin h-5 w-5 ms-3 bg-violet-700 rounded" viewBox="24 24 24 24" />
            </div>
        );
    }

    return (
        <div className="flex h-full items-center justify-center">
            <div className="grid grid-cols-1 md:grid-cols-2 gap-2 w-full">
                <div>
                    <div className="p-6 rounded bg-white shadow border">
                        <p className="text-xl font-sans text-black font-medium">Inference Session Resnet-50</p>
                        <p className="text-base font-sans text-black font-light my-2">
                            performs a classification task to predict whether an image contains the label cat or dog using the resnet-50 model
                            which is fine-tuned using the cat-and-dogs dataset which can be downloaded on kaggle
                        </p>
                        <input 
                            onChange={addImage}
                            accept="image/*"
                            type="file" 
                            className="block w-full text-sm text-slate-500 
                            file:mr-4 file:py-2 file:px-4 
                            file:rounded-full file:border-0 
                            file:text-sm file:font-semibold 
                            file:bg-violet-50 
                            file:text-violet-700 
                            hover:file:bg-violet-100
                        "/>
                        {image &&
                            <img className="my-4 object-contain h-64 w-full" src={image} />
                        }
                    </div>
                </div>

                <div>
                    <div className="p-6 rounded bg-white shadow border">
                        <p className="text-xl font-sans text-black font-medium">Prediction</p>
                        <p className="text-base font-sans text-black font-light my-2">
                            displays predictions from images that have been uploaded, there are 2 class names, namely cat and dog
                        </p>
                        <p className={prob && prob[0] > prob[1] ? "text-xl font-medium font-sans text-black my-2"  : "text-base font-sans text-black font-light my-2"}>
                            Cat : {prob ? (prob[0] * 100).toFixed(6) + "%" : "0%"}
                        </p>
                        <p className={prob && prob[1] > prob[0] ? "text-xl font-medium font-sans text-black my-2"  : "text-base font-sans text-black font-light my-2"}>
                            Dog : {prob ? (prob[1] * 100).toFixed(6) + "%" : "0%"}
                        </p>

                        {image &&
                            <img className="my-4 object-contain h-64 w-full" src={image} />
                        }
                    </div>
                </div>

            </div>
        </div>
    );
}
