// Tauri shell for the portfolio dashboard.
//
// On startup we spawn the bundled PyInstaller sidecar
// `dashboard-api`, capture the first line printed to stdout
// (which is the port the sidecar bound) and expose it to the
// WebView via the `get_api_port` IPC command. The frontend
// builds its `API_BASE` URL from that value on app load.
//
// The sidecar's lifecycle is tied to the parent: the
// `CommandChild` is stored in tauri::State and killed when the
// window is closed.

#![cfg_attr(
    all(not(debug_assertions), target_os = "windows"),
    windows_subsystem = "windows"
)]

use std::sync::Mutex;
use tauri::api::process::{Command, CommandChild, CommandEvent};
use tauri::{Manager, RunEvent, WindowEvent};

struct SidecarState {
    port: Mutex<Option<u16>>,
    child: Mutex<Option<CommandChild>>,
}

#[tauri::command]
fn get_api_port(state: tauri::State<SidecarState>) -> Option<u16> {
    *state.port.lock().unwrap()
}

fn main() {
    let state = SidecarState {
        port: Mutex::new(None),
        child: Mutex::new(None),
    };

    let app = tauri::Builder::default()
        .manage(state)
        .invoke_handler(tauri::generate_handler![get_api_port])
        .setup(|app| {
            let handle = app.handle();
            let (mut rx, child) = Command::new_sidecar("dashboard-api")
                .expect("dashboard-api sidecar must be bundled")
                .spawn()
                .expect("failed to spawn dashboard-api sidecar");

            // Store the child so it can be killed on exit.
            let state = handle.state::<SidecarState>();
            *state.child.lock().unwrap() = Some(child);

            // Read the port from the sidecar's first stdout line.
            let handle_clone = handle.clone();
            tauri::async_runtime::spawn(async move {
                while let Some(event) = rx.recv().await {
                    if let CommandEvent::Stdout(line) = event {
                        let trimmed = line.trim();
                        if let Ok(port) = trimmed.parse::<u16>() {
                            let st = handle_clone.state::<SidecarState>();
                            *st.port.lock().unwrap() = Some(port);
                            // Push the port to the WebView too,
                            // so `useApiPort` does not need to
                            // poll on first paint.
                            let _ = handle_clone.emit_all(
                                "api-port-ready",
                                serde_json::json!({ "port": port }),
                            );
                            break;
                        }
                    }
                }
            });
            Ok(())
        })
        .build(tauri::generate_context!())
        .expect("failed to build tauri app");

    app.run(|handle, event| {
        if let RunEvent::WindowEvent {
            event: WindowEvent::CloseRequested { .. },
            ..
        } = event
        {
            let state = handle.state::<SidecarState>();
            if let Some(child) = state.child.lock().unwrap().take() {
                let _ = child.kill();
            }
        }
    });
}
