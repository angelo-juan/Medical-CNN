//
//  ContentView.swift
//  medical ui
//
//  Created by 阮士榮 on 2025/1/8.
//

import SwiftUI
import AppKit

struct LogoView: View {
    var body: some View {
        ZStack {
            // 外圈
            Circle()
                .stroke(
                    LinearGradient(
                        gradient: Gradient(colors: [
                            Color(NSColor(red: 0.2, green: 0.6, blue: 0.9, alpha: 1)),
                            Color(NSColor(red: 0.4, green: 0.7, blue: 1.0, alpha: 1))
                        ]),
                        startPoint: .topLeading,
                        endPoint: .bottomTrailing
                    ),
                    lineWidth: 3
                )
                .frame(width: 60, height: 60)
            
            // 内部医疗十字
            Image(systemName: "cross.circle.fill")
                .font(.system(size: 40))
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
            
            // 波形图案
            Image(systemName: "waveform.path.ecg")
                .font(.system(size: 20))
                .foregroundColor(.white)
                .offset(y: 5)
        }
    }
}

struct ContentView: View {
    @State private var inputImage: NSImage?
    @State private var processedImage: NSImage?
    @State private var isShowingImagePicker = false
    @State private var isProcessing = false
    
    var body: some View {
        ZStack {
            // 背景渐变
            LinearGradient(gradient: Gradient(colors: [Color(NSColor(red: 0.95, green: 0.95, blue: 0.97, alpha: 1)), Color.white]), startPoint: .top, endPoint: .bottom)
                .ignoresSafeArea()
            
            VStack(spacing: 20) {
                // 添加图标和标题组合
                VStack(spacing: 15) {
                    LogoView()
                        .shadow(color: Color.black.opacity(0.1), radius: 5, x: 0, y: 2)
                    
                    Text("醫學影像處理系統")
                        .font(.system(size: 32, weight: .bold))
                        .foregroundColor(Color(NSColor(red: 0.2, green: 0.2, blue: 0.3, alpha: 1)))
                }
                .padding(.top, 20)
                
                // 主要内容区域
                HStack(spacing: 30) {
                    // 左侧：输入图像
                    VStack(spacing: 15) {
                        Text("原始影像")
                            .font(.system(size: 18, weight: .semibold))
                            .foregroundColor(Color(NSColor(red: 0.3, green: 0.3, blue: 0.4, alpha: 1)))
                        
                        ZStack {
                            RoundedRectangle(cornerRadius: 15)
                                .fill(Color.white)
                                .shadow(color: Color.black.opacity(0.1), radius: 5, x: 0, y: 2)
                            
                            if let image = inputImage {
                                Image(nsImage: image)
                                    .resizable()
                                    .scaledToFit()
                                    .cornerRadius(12)
                                    .padding(8)
                            } else {
                                VStack(spacing: 15) {
                                    Image(systemName: "photo.on.rectangle.angled")
                                        .font(.system(size: 40))
                                        .foregroundColor(Color(NSColor(red: 0.6, green: 0.6, blue: 0.7, alpha: 1)))
                                    
                                    Text("點擊選擇圖片")
                                        .font(.system(size: 16))
                                        .foregroundColor(Color(NSColor(red: 0.5, green: 0.5, blue: 0.6, alpha: 1)))
                                }
                            }
                        }
                        .frame(width: 300, height: 300)
                        .onTapGesture {
                            openImagePicker()
                        }
                    }
                    
                    // 中间：箭头
                    VStack {
                        if isProcessing {
                            ProgressView()
                                .scaleEffect(1.5)
                                .padding()
                        } else {
                            Image(systemName: "arrow.right.circle.fill")
                                .font(.system(size: 36))
                                .foregroundColor(Color(NSColor(red: 0.3, green: 0.6, blue: 0.9, alpha: 1)))
                        }
                    }
                    
                    // 右侧：处理后图像
                    VStack(spacing: 15) {
                        Text("處理結果")
                            .font(.system(size: 18, weight: .semibold))
                            .foregroundColor(Color(NSColor(red: 0.3, green: 0.3, blue: 0.4, alpha: 1)))
                        
                        ZStack {
                            RoundedRectangle(cornerRadius: 15)
                                .fill(Color.white)
                                .shadow(color: Color.black.opacity(0.1), radius: 5, x: 0, y: 2)
                            
                            if let processed = processedImage {
                                Image(nsImage: processed)
                                    .resizable()
                                    .scaledToFit()
                                    .cornerRadius(12)
                                    .padding(8)
                            } else {
                                VStack(spacing: 15) {
                                    Image(systemName: "waveform.path.ecg")
                                        .font(.system(size: 40))
                                        .foregroundColor(Color(NSColor(red: 0.6, green: 0.6, blue: 0.7, alpha: 1)))
                                    
                                    Text("等待處理")
                                        .font(.system(size: 16))
                                        .foregroundColor(Color(NSColor(red: 0.5, green: 0.5, blue: 0.6, alpha: 1)))
                                }
                            }
                        }
                        .frame(width: 300, height: 300)
                    }
                }
                .padding(.horizontal)
                
                // 处理按钮
                Button(action: {
                    isProcessing = true
                    DispatchQueue.main.asyncAfter(deadline: .now() + 0.5) {
                        processImage()
                        isProcessing = false
                    }
                }) {
                    HStack {
                        Image(systemName: "wand.and.stars")
                        Text("開始處理")
                    }
                    .font(.system(size: 16, weight: .medium))
                    .foregroundColor(.white)
                    .padding(.horizontal, 50)
                    .padding(.vertical, 12)
                    .background(inputImage != nil ? 
                        LinearGradient(gradient: Gradient(colors: [Color(NSColor(red: 0.3, green: 0.6, blue: 0.9, alpha: 1)), Color(NSColor(red: 0.4, green: 0.7, blue: 1.0, alpha: 1))]), startPoint: .leading, endPoint: .trailing) :
                        LinearGradient(gradient: Gradient(colors: [Color.gray, Color.gray]), startPoint: .leading, endPoint: .trailing))
                    .cornerRadius(25)
                    .shadow(color: Color.black.opacity(0.1), radius: 3, x: 0, y: 2)
                }
                .disabled(inputImage == nil || isProcessing)
                .padding(.top, 20)
            }
            .padding()
        }
        .frame(minWidth: 800, minHeight: 700)
    }
    
    func openImagePicker() {
        let panel = NSOpenPanel()
        panel.allowsMultipleSelection = false
        panel.canChooseDirectories = false
        panel.canChooseFiles = true
        panel.allowedContentTypes = [.image]
        
        panel.beginSheetModal(for: NSApp.mainWindow ?? NSWindow()) { response in
            if response == .OK, let url = panel.url {
                inputImage = NSImage(contentsOf: url)
            }
        }
    }
    
    func processImage() {
        // TODO: 在这里添加CNN模型处理逻辑
        processedImage = inputImage
    }
}

#Preview {
    ContentView()
}
