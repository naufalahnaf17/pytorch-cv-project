'use server'

import sharp from "sharp"

export async function imagePreprocess(image){
    let image_raw = new Buffer.from(image,'base64')
    let data = await sharp(image_raw)
        .removeAlpha()
        .resize(224,224)
        .raw()
        .toBuffer()

    const [red,green,blue] = new Array(new Array(),new Array(),new Array())
    for(let i = 0;i < data.length; i+=3){
        red.push(data[i] / 255.0)
        green.push(data[i + 1] / 255.0)
        blue.push(data[i + 2] / 255.0)
    }

    let imageTransposed = red.concat(green).concat(blue)
    return imageTransposed
}

export async function softmaxPrediction(arr){
    const expScores = arr.map(Math.exp);
    const sumExpScores = expScores.reduce((a, b) => a + b, 0);
    const softmaxScores = expScores.map(score => score / sumExpScores);
    const softmaxWithIndex = softmaxScores.map((value, index) => ({ value, index }));
    const sortedSoftmax = softmaxWithIndex.sort((a, b) => b.value - a.value);
    return sortedSoftmax.slice(0, 5); 
}