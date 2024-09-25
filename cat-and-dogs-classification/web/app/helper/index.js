'use server'

import sharp from "sharp";

export async function resizeImage(image) {
    let bufferImage = new Buffer.from(image,'base64')
    const { data, info } = await sharp(bufferImage)
        .removeAlpha()
        .resize(224, 224)
        .raw()
        .toBuffer({ resolveWithObject: true });

    return data;
}