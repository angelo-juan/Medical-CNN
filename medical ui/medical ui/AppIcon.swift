import SwiftUI
import AppKit

struct AppIcon: View {
    var body: some View {
        ZStack {
            // 背景渐变
            LinearGradient(
                gradient: Gradient(colors: [
                    Color(NSColor(red: 0.2, green: 0.6, blue: 0.9, alpha: 1)),
                    Color(NSColor(red: 0.4, green: 0.7, blue: 1.0, alpha: 1))
                ]),
                startPoint: .topLeading,
                endPoint: .bottomTrailing
            )
            
            // 白色圆形背景
            Circle()
                .fill(.white)
                .padding(20)
            
            // 医疗十字
            Image(systemName: "cross.case.fill")
                .font(.system(size: 150))
                .foregroundStyle(
                    LinearGradient(
                        gradient: Gradient(colors: [
                            Color(NSColor(red: 0.2, green: 0.6, blue: 0.9, alpha: 1)),
                            Color(NSColor(red: 0.4, green: 0.7, blue: 1.0, alpha: 1))
                        ]),
                        startPoint: .topLeading,
                        endPoint: .bottomTrailing
                    )
                )
                .padding(40)
            
            // 波形图案
            Image(systemName: "waveform.path.ecg")
                .font(.system(size: 80))
                .foregroundColor(.white)
                .offset(y: 20)
                .shadow(color: Color.black.opacity(0.2), radius: 2, x: 0, y: 2)
        }
        .frame(width: 512, height: 512) // 最大尺寸
        .background(Color(NSColor(red: 0.2, green: 0.6, blue: 0.9, alpha: 1)))
        .clipShape(RoundedRectangle(cornerRadius: 100))
    }
    
    static func exportIcon() {
        let sizes = [16, 32, 64, 128, 256, 512, 1024]
        let view = AppIcon()
        
        // 获取下载文件夹路径
        let downloadsPath = FileManager.default.homeDirectoryForCurrentUser
            .appendingPathComponent("Downloads")
            .appendingPathComponent("AppIcon.iconset")
        
        do {
            // 创建iconset文件夹
            try FileManager.default.createDirectory(at: downloadsPath, withIntermediateDirectories: true)
            
            // 导出所有尺寸的图标
            for size in sizes {
                let rect = CGRect(x: 0, y: 0, width: size, height: size)
                let hosting = NSHostingView(rootView: view.frame(width: CGFloat(size), height: CGFloat(size)))
                hosting.frame = rect
                
                // 创建图像表示
                let image = NSImage(size: rect.size)
                image.lockFocus()
                
                if let context = NSGraphicsContext.current {
                    hosting.layer?.render(in: context.cgContext)
                }
                
                image.unlockFocus()
                
                // 保存常规尺寸
                if let tiffData = image.tiffRepresentation,
                   let bitmap = NSBitmapImageRep(data: tiffData),
                   let data = bitmap.representation(using: .png, properties: [:]) {
                    let filename = "icon_\(size)x\(size).png"
                    try data.write(to: downloadsPath.appendingPathComponent(filename))
                }
                
                // 保存 @2x 版本
                if size <= 512 {
                    let doubleSize = size * 2
                    let doubleRect = CGRect(x: 0, y: 0, width: doubleSize, height: doubleSize)
                    let doubleHosting = NSHostingView(rootView: view.frame(width: CGFloat(doubleSize), height: CGFloat(doubleSize)))
                    doubleHosting.frame = doubleRect
                    
                    let doubleImage = NSImage(size: doubleRect.size)
                    doubleImage.lockFocus()
                    
                    if let context = NSGraphicsContext.current {
                        doubleHosting.layer?.render(in: context.cgContext)
                    }
                    
                    doubleImage.unlockFocus()
                    
                    if let tiffData = doubleImage.tiffRepresentation,
                       let bitmap = NSBitmapImageRep(data: tiffData),
                       let data = bitmap.representation(using: .png, properties: [:]) {
                        let filename = "icon_\(size)x\(size)@2x.png"
                        try data.write(to: downloadsPath.appendingPathComponent(filename))
                    }
                }
            }
            
            // 转换为 .icns 文件
            let task = Process()
            task.launchPath = "/usr/bin/iconutil"
            task.arguments = ["-c", "icns", downloadsPath.path]
            try task.run()
            task.waitUntilExit()
            
            print("图标已成功导出到下载文件夹")
        } catch {
            print("导出图标时发生错误：\(error)")
        }
    }
}

#Preview {
    VStack {
        AppIcon()
            .frame(width: 512, height: 512)
        
        Button("导出图标") {
            AppIcon.exportIcon()
        }
        .padding()
    }
} 