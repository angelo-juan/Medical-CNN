//
//  medical_uiApp.swift
//  medical ui
//
//  Created by 阮士榮 on 2025/1/8.
//

import SwiftUI

@main
struct medical_uiApp: App {
    var body: some Scene {
        WindowGroup {
            ContentView()
        }
        .commands {
            // 添加自定義菜單項
            CommandGroup(replacing: .appInfo) {
                Button("關於醫學影像處理系統") {
                    NSApplication.shared.orderFrontStandardAboutPanel()
                }
            }
        }
        .windowStyle(.hiddenTitleBar)
        .windowResizability(.contentSize)
        .defaultSize(width: 800, height: 700) // 設置默認窗口大小
    }
}
