
class Classify extends tf.layers.Layer {
    constructor() {
        super({});
    }
    computeOutputShape(inputShape) { return inputShape; }
    build() {}
    call(input) {
        return tf.tidy(() => {
            return tf.concat([
                tf.sigmoid(input[0].slice([0, 0, 0, 0], [-1, -1, -1, 1])),
                tf.relu(input[0].slice([0, 0, 0, 1], [-1, -1, -1, -1]))
            ], 3);
        });
    }
    getConfig() { return super.getConfig(); }
    static get className() { return "Classify"; }
}
class Skip extends tf.layers.Layer {
    constructor() {
        super({});
    }
    computeOutputShape(inputShape) { return [null, inputShape[1][1], inputShape[1][2], inputShape[0][3] + inputShape[1][3]]; }
    build() {}
    call(input) {
        return tf.tidy(() => {
            return tf.concat([
                input[1],
                input[0].slice([0, 0, 0, 0], input[1].shape)
            ], 3);
        });
    }
    getConfig() { return super.getConfig(); }
    static get className() { return "Skip"; }
}
tf.serialization.registerClass(Classify);
tf.serialization.registerClass(Skip);

function loss(t, p) {
    return tf.tidy(() => {
        const [batchSize, height, width, channel] = p.shape;
        // cross-entropy loss on all binary classification; mean squared error on position predictions of true text
        const binaryLabels = t.slice([0, 0, 0, 0], [-1, -1, -1, 1]);
        const binaryPredictions = p.slice([0, 0, 0, 0], [-1, -1, -1, 1]);
        const crossEntropy = tf.metrics.binaryCrossentropy(binaryLabels, binaryPredictions);
        const posLabels = t.slice([0, 0, 0, 1]);
        const posPredictions = p.slice([0, 0, 0, 1]);
        // const newPos = posPredictions.where(binaryLabels.cast("bool"), posLabels);
        const newPos = posPredictions.mul(binaryLabels);
        // console.log(posPredictions.shape, binaryLabels.shape, posLabels.shape, newPos.shape);
        // const posError = tf.losses.meanSquaredError(posPredictions.mul(binaryLabels), posLabels.mul(binaryLabels));
        const posError = tf.losses.meanSquaredError(newPos, posLabels);
        return crossEntropy.mean().add(posError);
        // return posError;
    });
}

function createModel() {
    // model = tf.sequential();
    // model.add(tf.layers.conv2d({
    //     inputShape: [null, null, 1],
    //     filters: 16,
    //     kernelSize: 7,
    //     activation: "relu",
    //     padding: "same"
    // }));
    // model.add(tf.layers.conv2d({
    //     filters: 16,
    //     kernelSize: 3,
    //     activation: "relu",
    //     padding: "same"
    // }));
    // model.add(tf.layers.conv2d({
    //     filters: 5,
    //     kernelSize: 1,
    //     padding: "same"
    // }));
    // model.add(new Classify());
    const inputs = tf.input({ shape: [null, null, 1]});
    const architecture = [8, 16];
    let layer = inputs;
    const layers = [];
    for(let i = 0; i < architecture.length; i += 1) {
        layer = tf.layers.conv2d({
            filters: architecture[i],
            kernelSize: 3,
            activation: "relu",
            padding: "same"
        }).apply(layer);
        layer = tf.layers.dropout({ rate: 0.1 }).apply(layer);
        layer = tf.layers.conv2d({
            filters: architecture[i],
            kernelSize: 3,
            activation: "relu",
            padding: "same"
        }).apply(layer);
        layers.push(layer);
        layer = tf.layers.batchNormalization().apply(layer);
        layer = tf.layers.reLU().apply(layer);
        layer = tf.layers.maxPooling2d({ poolSize: 2 }).apply(layer);
    }
    layer = tf.layers.conv2d({
        filters: 32,
        kernelSize: 3,
        activation: "relu",
        padding: "same"
    }).apply(layer);
    layer = tf.layers.batchNormalization().apply(layer);
    layer = tf.layers.reLU().apply(layer);
    layer = tf.layers.dropout({ rate: 0.1 }).apply(layer);
    layer = tf.layers.conv2d({
        filters: 32,
        kernelSize: 3,
        activation: "relu",
        padding: "same"
    }).apply(layer);
    for(let i = architecture.length - 1; i >= 0; i -= 1) {
        layer = tf.layers.conv2dTranspose({
            filters: architecture[i],
            kernelSize: 2,
            strides: 2,
            padding: "same"
        }).apply(layer);
        layer = new Skip().apply([layers[i], layer]);
        // layer = tf.layers.concatenate().apply([layer, layers[i]]);
        layer = tf.layers.batchNormalization().apply(layer);
        layer = tf.layers.reLU().apply(layer);
    }
    layer = tf.layers.conv2d({
        filters: 5,
        kernelSize: 1,
        padding: "same"
    }).apply(layer);
    const outputs = new Classify().apply(layer);

    model = tf.model({ inputs, outputs });
    return model;
}
var model = createModel();

function makeIteratorMaker(batchSize) {
    let batchNum = 0;
    return function() {
        return {
            next: () => {
                return tf.tidy(() => {
                    const xs = [], ys = [];
                    for(let i = 0; i < batchSize; i += 1) {
                        [x, y] = sample();
                        xs.push(x);
                        ys.push(y);
                    }
                    console.log("batch generated");
                    [testImage, testMask] = sample();
                    test();
                    return { value: { xs: tf.stack(xs), ys: tf.stack(ys) }, done: false };
                });
            }
        };
    };
}

var num = 0;
function train(batchesPerEpoch = +document.getElementById("batches").value, epochs = +document.getElementById("epochs").value, batchSize = +document.getElementById("batchSize").value) {
    console.log("preparing data");
    const dataset = tf.data.generator(makeIteratorMaker(batchSize));
    console.log("train start");
    const optimizer = tf.train.adam();
    console.log("initalizing");
    model.compile({
        optimizer,
        loss,
        metrics: ["accuracy"]
    });
    console.log("compiled");
    model.fitDataset(dataset, { batchesPerEpoch, epochs }).then(info => {
        console.log("accuracy", info.history.acc);
        optimizer.dispose();
        const newNum = tf.memory().numTensors;
        console.log(`${num} + ${newNum - num} = ${newNum}`);
        num = newNum;
    });
    // return model.fit(images, masks, {
    //     epochs,
    //     batchSize,
    //     callbacks: {
    //         onBatchEnd: (batch, logs) => console.log("accuracy", logs.acc)
    //     }
    // }).then(info => {
}

async function save() {
    await model.save("downloads://text");
    console.log("saved");
}

async function load() {
    console.log("loading");
    const newModel = await tf.loadLayersModel(tf.io.browserFiles([
        document.getElementById("json").files[0],
        document.getElementById("weights").files[0]
    ]));
    for(let i = 0; i < newModel.layers.length; i += 1) {
    // for(let i = 0; i < 26; i += 1) {
        model.layers[i].setWeights(newModel.layers[i].getWeights());
    }
    newModel.dispose();
    console.log("loaded");
}

// document.addEventListener("DOMContentLoaded", async function () {
//     // Load and preprocess the MNIST dataset
//     const dataset = await tf.data.mnist();
//     const trainData = dataset.trainImages.reshape([-1, 28, 28, 1]);
//     const testData = dataset.testImages.reshape([-1, 28, 28, 1]);
//     const trainLabels = tf.oneHot(dataset.trainLabels, 10);
//     const testLabels = tf.oneHot(dataset.testLabels, 10);

//     // Build the fully convolutional neural network (FCNN) model
//     const model = tf.sequential();
//     model.add(tf.layers.conv2d({
//         inputShape: [28, 28, 1],
//         filters: 32,
//         kernelSize: 3,
//         activation: 'relu'
//     }));
//     model.add(tf.layers.maxPooling2d({ poolSize: 2, strides: 2 }));
//     model.add(tf.layers.flatten());
//     model.add(tf.layers.dense({ units: 128, activation: 'relu' }));
//     model.add(tf.layers.dense({ units: 10, activation: 'softmax' }));

//     // Compile the model
//     model.compile({
//         optimizer: 'adam',
//         loss: 'categoricalCrossentropy',
//         metrics: ['accuracy']
//     });

//     // Train the model
//     await model.fit(trainData, trainLabels, {
//         epochs: 5,
//         batchSize: 64,
//         validationData: [testData, testLabels]
//     });

//     // Evaluate the model
//     const evalResult = await model.evaluate(testData, testLabels);
//     console.log('Evaluation Result:', evalResult);

//     // Perform inference on a sample image
//     const inferenceResult = model.predict(testData.slice([0, 0, 0, 0], [1, 28, 28, 1]));
//     const predictedClass = inferenceResult.argMax(1).dataSync()[0];
//     console.log('Predicted Class:', predictedClass);
// });