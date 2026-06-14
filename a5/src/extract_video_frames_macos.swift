import AVFoundation
import Darwin
import Foundation
import ImageIO
import UniformTypeIdentifiers

let arguments = CommandLine.arguments
let sourcePath = arguments.count > 1 ? arguments[1] : "a5/video/IMG_6753.MOV"
let outputPath = arguments.count > 2 ? arguments[2] : "a5/data/scene/images"
let interval = arguments.count > 3 ? Double(arguments[3]) ?? 0.5 : 0.5
let maxFrames = arguments.count > 4 ? Int(arguments[4]) ?? 40 : 40

guard interval > 0, maxFrames > 0 else {
    fatalError("Interval and maxFrames must be positive")
}

let sourceURL = URL(fileURLWithPath: sourcePath)
let outputURL = URL(fileURLWithPath: outputPath, isDirectory: true)
guard FileManager.default.fileExists(atPath: sourceURL.path) else {
    fatalError("Source video does not exist: \(sourceURL.path)")
}

try FileManager.default.createDirectory(at: outputURL, withIntermediateDirectories: true)
for file in try FileManager.default.contentsOfDirectory(at: outputURL, includingPropertiesForKeys: nil) {
    if file.lastPathComponent.hasPrefix("frame_") && file.pathExtension.lowercased() == "jpg" {
        try FileManager.default.removeItem(at: file)
    }
}

let asset = AVURLAsset(url: sourceURL)
let durationSeconds = CMTimeGetSeconds(asset.duration)
let generator = AVAssetImageGenerator(asset: asset)
generator.appliesPreferredTrackTransform = true
generator.maximumSize = CGSize(width: 1600, height: 1600)
generator.requestedTimeToleranceBefore = CMTime(value: 1, timescale: 20)
generator.requestedTimeToleranceAfter = CMTime(value: 1, timescale: 20)

var frames: [[String: Any]] = []
for index in 0..<maxFrames {
    let requestedSeconds = Double(index) * interval
    if requestedSeconds > durationSeconds {
        break
    }

    let requestedTime = CMTime(seconds: requestedSeconds, preferredTimescale: 600)
    var actualTime = CMTime.zero
    let image: CGImage
    do {
        image = try generator.copyCGImage(at: requestedTime, actualTime: &actualTime)
    } catch {
        print("Skipped \(String(format: "%.2f", requestedSeconds))s: \(error)")
        continue
    }
    let actualSeconds = CMTimeGetSeconds(actualTime)
    let filename = String(format: "frame_%04d_%08.2fs.jpg", index, actualSeconds)
    let destinationURL = outputURL.appendingPathComponent(filename)

    guard let destination = CGImageDestinationCreateWithURL(
        destinationURL as CFURL,
        UTType.jpeg.identifier as CFString,
        1,
        nil
    ) else {
        fatalError("Could not create JPEG destination: \(destinationURL.path)")
    }
    let options = [kCGImageDestinationLossyCompressionQuality: 0.95] as CFDictionary
    CGImageDestinationAddImage(destination, image, options)
    guard CGImageDestinationFinalize(destination) else {
        fatalError("Could not write JPEG: \(destinationURL.path)")
    }

    frames.append([
        "file": filename,
        "requested_time_seconds": requestedSeconds,
        "actual_time_seconds": actualSeconds,
        "width": image.width,
        "height": image.height,
    ])
    print("Extracted \(filename)")
}

if frames.count < 3 {
    fputs("Only \(frames.count) decodable frames were extracted. Re-export or replace the video.\n", stderr)
    exit(1)
}

let manifest: [String: Any] = [
    "source_video": sourcePath,
    "source_duration_seconds": durationSeconds,
    "decoder_backend": "AVFoundation",
    "interval_seconds": interval,
    "exported_frame_count": frames.count,
    "frames": frames,
]
let manifestURL = outputURL.deletingLastPathComponent().appendingPathComponent("video_frames.json")
let manifestData = try JSONSerialization.data(withJSONObject: manifest, options: [.prettyPrinted, .sortedKeys])
try manifestData.write(to: manifestURL)

print("Extracted \(frames.count) frames to \(outputURL.path)")
print("Manifest: \(manifestURL.path)")
