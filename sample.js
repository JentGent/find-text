
function line(x1, y1, x2, y2) {
    ctx.beginPath();
    ctx.moveTo(x1, y1);
    ctx.lineTo(x2, y2);
    ctx.stroke();
    ctx.closePath();
}

function sample() {
    const fonts = ["Arial", "Verdana", "Helvetica", "Times New Roman", "Courier New", "Georgia", "Comic Sans MS", "Trebuchet MS"];
    // const width = random(200, 400) | 0,
    //       height = random(100, 300) | 0;
    const width = 200,
          height = 100;
    cnv.width = width;
    cnv.height = height;
    ctx.fillStyle = `rgb(${random(255)}, ${random(255)}, ${random(255)})`;
    ctx.fillRect(0, 0, width, height);
    for(let i = 0; i < 50; i += 1) {
        ctx.fillStyle = `rgb(${random(255)}, ${random(255)}, ${random(255)})`;
        ctx.strokeStyle = `rgb(${random(255)}, ${random(255)}, ${random(255)})`;
        ctx.lineWidth = random(1, 10);
        switch(random(3) | 0) {
            case 0: ctx.fillRect(random(width), random(height), random(100), random(100));
            break;
            case 1: line(random(width), random(height), random(width), random(height));
            break;
            case 2:
                ctx.beginPath();
                ctx.ellipse(random(width), random(height), random(100), random(100), random(Math.PI), 0, Math.PI * 2);
                ctx.fill();
        }
    }
    const image = Array.from({ length: height }, () => Array(width).fill(0)),
        mask = Array.from({ length: height }, () => Array.from({ length: width }, () => Array(5).fill(0)));
        //   mask = Array.from({ length: 5 }, () => Array.from({ length: height }, () => Array(width).fill(0)));
    const boxes = random() < 0.5;
    for(let j = 0; j < (boxes ? 1 : random(5)); j += 1) {
        ctx.fillStyle = `rgb(${random(255)}, ${random(255)}, ${random(255)})`;
        let text = "";
        for(let i = 0; i < random(50); i += 1) {
            text += characters[random(characters.length) | 0];
        }
        const size = random(20, 40),
              font = fonts[random(fonts.length) | 0],
              spacing = random(size / 5) | 0;
        ctx.font = `${size}px ${font}`;
        const measure = ctx.measureText(text);
        let x = random(width - measure.width) | 0,
            y = random(measure.actualBoundingBoxAscent, height - measure.actualBoundingBoxDescent) | 0;
        if(boxes) {
            const w = random() < 0.5;
            ctx.fillStyle = w ? "black" : "white";
            ctx.fillRect(x - 20, y - measure.actualBoundingBoxAscent - 20, measure.width + spacing * text.length + 40, measure.actualBoundingBoxAscent + measure.actualBoundingBoxDescent + 40);
            ctx.fillStyle = w ? "white" : "black";
        }
        for(let i = 0; i < text.length; i += 1) {
            const measure = ctx.measureText(text[i]);
            const top = Math.ceil(measure.fontBoundingBoxAscent),
                  bottom = Math.ceil(measure.fontBoundingBoxDescent),
                  right = Math.ceil(measure.width);
            // line(x, y - top, x + right, y + bottom);
            ctx.fillText(text[i], x, y);
            msk.width = right;
            msk.height = top + bottom;
            mctx.clearRect(0, 0, msk.width, msk.height);
            mctx.font = ctx.font;
            mctx.fillText(text[i], 0, top);
            const data = mctx.getImageData(0, 0, msk.width, msk.height).data;
            let px = x, py = y - top;
            for(let k = 0; k < data.length; k += 4) {
                if(data[k + 4 - 1] !== 0 && px >= 0 && px < width && py >= 0 && py < height) {
                    mask[py][px][0] = 1;
                    mask[py][px][1] = (px - x) / 10;
                    mask[py][px][2] = (py - (y - top)) / 10;
                    mask[py][px][3] = ((x + right) - px) / 10;
                    mask[py][px][4] = ((y + bottom) - py) / 10;
                }
                px += 1;
                if(px >= x + msk.width) {
                    px = x;
                    py += 1;
                }
            }
            x += right + spacing;
        }
    }
    const data = ctx.getImageData(0, 0, width, height).data;
    let px = 0, py = 0;
    for(let i = 0; i < data.length; i += 4) {
        image[py][px] = [(data[i] + data[i + 1] + data[i + 2]) / 3];
        ctx.fillStyle = `rgb(${image[py][px]}, ${image[py][px]}, ${image[py][px]})`;
        ctx.fillRect(px, py, 1, 1);
        px += 1;
        if(px >= width) {
            px = 0;
            py += 1;
        }
    }
    return [tf.tensor(image), tf.tensor(mask)];
}

function image() {
    const image = new Image();
    const reader = new FileReader();
    reader.onload = function(e) {
        image.src = e.target.result;
    };
    reader.readAsDataURL(document.getElementById("upload").files[0]);
    image.onload = function() {
        img.width = image.width;
        img.height = image.height;
        ictx.drawImage(image, 0, 0);
        const imageData = ictx.getImageData(0, 0, img.width, img.height);
        const data = imageData.data;
        const input = [];
        for(let i = 0; i < data.length; i += 4) {
            const c = (data[i] + data[i + 1] + data[i + 2]) / 3 * data[i + 3] / 255;
            data[i] = data[i + 1] = data[i + 2] = c;
            data[i + 3] = 255;
            input.push(c);
        }
        ictx.putImageData(imageData, 0, 0);
        prd.width = img.width;
        prd.height = img.height;
        test(img, ictx, prd, pctx, tf.tensor(input, [1, img.height, img.width, 1]));
    }
}

function test(cnv = cnv, ctx = ctx, tst = tst, tctx = tctx, input = testImage.reshape([1, 100, 200, 1])) {
    tf.tidy(() => {
        tst.width = cnv.width;
        tst.height = cnv.height;
        const prediction = model.predict(input).arraySync()[0];
        // const prediction = testMask.arraySync();
        // console.log(prediction);
        const threshold = document.getElementById("threshold").value / 100;
        for(let y = 0; y < prediction.length; y += 1) {
            for(let x = 0; x < prediction[y].length; x += 1) {
                const [text, left, top, right, bottom] = prediction[y][x];
                const c = text * 255;
                tctx.fillStyle = text > threshold ? "green" : `rgb(${c}, ${c}, ${c})`;
                tctx.fillRect(x, y, 1, 1);
            }
        }
        for(let y = 0; y < prediction.length; y += 1) {
            for(let x = 0; x < prediction[y].length; x += 1) {
                const [text, left, top, right, bottom] = prediction[y][x];
                if(text > threshold) {
                    tctx.fillStyle = "red";
                    tctx.fillRect(x - left * 10, y - top * 10, 1, 1);
                    tctx.fillStyle = "cyan";
                    tctx.fillRect(x + right * 10, y + bottom * 10, 1, 1);
                }
            }
        }
    });
}
