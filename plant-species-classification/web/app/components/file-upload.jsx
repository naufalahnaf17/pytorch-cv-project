'use client'

import { useState,useEffect } from "react";
import * as ort from "onnxruntime-web"
import {imagePreprocess,softmaxPrediction} from "../../helpers/image-process"
import {getClassNames} from "../../helpers/class-names"
import LoadingComponent from "./loading";

export default function FileUpload() {
    const [session,setSession] = useState(null)
    const [image,setImage] = useState(null)
    const [labels,setLabels] = useState(null)
    const [probalities,setProbabilities] = useState(null)
    const [loading,setLoading] = useState(true)
    const [loadingInference,setLoadingInference] = useState(false)

    useEffect(() => {
        setClassNames()
        createSession()
    },[])

    const setClassNames = async () => {
        let labels = await getClassNames()
        setLabels(labels)
    }

    const createSession = async () => {
        const session = await ort.InferenceSession.create("../plant-classification.onnx")
        setSession(session)
        setLoading(false)
    }

    const handleImage = async (e) => {
        setLoadingInference(true)
        setProbabilities(null)
        let files = e.target.files[0]
        let reader = new FileReader()
        reader.onload = (x) => {
            let base64 = x.target.result
            let base64Data = base64.split(',')[1]
            setImage(base64)
            createImageTensor(base64Data)
        }
        reader.readAsDataURL(files)
    }

    const createImageTensor = async (data) => {
        let dims = [1,3,224,224]
        let image = await imagePreprocess(data)
        let tensor = await new ort.Tensor('float32',image,dims)
        inferenceImage(tensor)
    }

    const inferenceImage = async (tensor) => {
        const pred = await session.run({ input : tensor })
        const prob = await softmaxPrediction(pred.output.data)

        let top5 = []
        prob.map((e) => {
            top5.push({
                prob : e.value,
                labels : labels[e.index]
            })
        })

        setProbabilities(top5)
        setLoadingInference(false)
    }

    if (loading){
        return(
            <div className="h-screen w-full flex flex-row items-center justify-center">
                <p className="text-3xl font-sans mr-3">Loading Models</p>
                <LoadingComponent />
            </div>
        )
    }

    return (
        <div className="flex h-full items-center justify-center">
            <div className="grid grid-cols-1 md:grid-cols-2 gap-2 w-full">
                <div className="flex w-full">
                    <div className="p-6 rounded bg-white shadow border">
                        <p className="text-xl font-sans text-black font-medium">Inference Session VIT_Base_Patch16_224</p>
                        <p className="text-base font-sans text-black font-light my-2">
                            performs a classification task to predict whether an image contains 47 label using the VIT_Base_Patch16_224 model
                            which is fine-tuned using plant-species dataset which can be downloaded on kaggle
                        </p>
                        <input 
                            onChange={handleImage}
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

                <div className="flex w-full">
                    <div className="p-6 rounded bg-white shadow border w-full">
                        <p className="text-xl font-sans text-black font-medium mb-4">Prediction</p>
                        {loadingInference &&
                            <div className="flex items-center justify-center md:justify-start">
                                <LoadingComponent />
                            </div>
                        }
                        {probalities &&
                            probalities.map((e,i) => {
                                return(
                                    <div key={i} className="flex flex-row justify-between">
                                        <p className="text-base font-medium">{e.labels}</p>
                                        <p className="text-base font-medium">{(e.prob * 100).toFixed(6) + "%"}</p>
                                    </div>
                                )
                            })
                        }
                    </div>
                </div>

            </div>
        </div>
    );
}