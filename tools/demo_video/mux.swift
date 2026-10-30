// usage: swift mux.swift video.mp4 audio.m4a out.mp4   (combines the two tracks without re-encoding the video)
import AVFoundation
import Foundation

let a = CommandLine.arguments
guard a.count == 4 else { print("usage: mux video audio out"); exit(2) }
let video = AVURLAsset(url: URL(fileURLWithPath: a[1]))
let audio = AVURLAsset(url: URL(fileURLWithPath: a[2]))
let comp = AVMutableComposition()
guard let vt = video.tracks(withMediaType: .video).first, let at = audio.tracks(withMediaType: .audio).first,
      let cv = comp.addMutableTrack(withMediaType: .video, preferredTrackID: kCMPersistentTrackID_Invalid),
      let ca = comp.addMutableTrack(withMediaType: .audio, preferredTrackID: kCMPersistentTrackID_Invalid)
else { print("missing tracks"); exit(1) }
try cv.insertTimeRange(CMTimeRange(start: .zero, duration: video.duration), of: vt, at: .zero)
let d = CMTimeMinimum(audio.duration, video.duration + CMTime(seconds: 1, preferredTimescale: 600))
try ca.insertTimeRange(CMTimeRange(start: .zero, duration: d), of: at, at: .zero)
try? FileManager.default.removeItem(atPath: a[3])
guard let ex = AVAssetExportSession(asset: comp, presetName: AVAssetExportPresetPassthrough) else { print("no exporter"); exit(1) }
ex.outputURL = URL(fileURLWithPath: a[3]); ex.outputFileType = .mp4
let sem = DispatchSemaphore(value: 0)
ex.exportAsynchronously { sem.signal() }
sem.wait()
if ex.status != .completed { print("export failed:", ex.error?.localizedDescription ?? "?"); exit(1) }
print("ok")
