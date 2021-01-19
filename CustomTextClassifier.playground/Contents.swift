import Foundation
import CreateML

let dataPath = "/Users/shailesh/Downloads/Reviews.json"

let rawData = URL(fileURLWithPath: dataPath)

let dataset = try MLDataTable(contentsOf: rawData)

let (trainingData, testData) = dataset.randomSplit(by: 0.8, seed: 7)

let model = try MLTextClassifier(trainingData: trainingData, textColumn: "text", labelColumn: "label")

let metrics = model.evaluation(on: testData, textColumn: "text", labelColumn: "label")

let accuracy = (1 - metrics.classificationError) * 100
let confusion = metrics.confusion

let modelPath = "/Users/shailesh/Downloads/ReviewMLTextClassifier.mlmodel"

let coreMLModel = URL(fileURLWithPath: modelPath)
try model.write(to: coreMLModel)

