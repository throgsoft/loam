use std::io::{BufRead, BufReader, Write};
use std::net::{TcpListener, TcpStream};
use std::path::Path;
use std::time::Duration;

pub fn run(root: &Path, port: u16) -> Result<(), String> {
    let listener = TcpListener::bind(("127.0.0.1", port))
        .map_err(|e| format!("bind 127.0.0.1:{port}: {e}"))?;
    println!("serving {} at http://127.0.0.1:{port}/", root.display());
    for stream in listener.incoming().flatten() {
        let root = root.to_path_buf();
        std::thread::spawn(move || respond(&root, stream));
    }
    Ok(())
}

fn respond(root: &Path, mut stream: TcpStream) -> std::io::Result<()> {
    stream.set_read_timeout(Some(Duration::from_secs(5)))?;
    let mut reader = BufReader::new(stream.try_clone()?);
    let mut line = String::new();
    reader.read_line(&mut line)?;
    let target = line.split_whitespace().nth(1).unwrap_or("/");
    let path = target.split('?').next().unwrap_or("/").to_string();
    let mut header = String::new();
    while reader.read_line(&mut header)? > 2 {
        header.clear();
    }
    let safe = |part: &&str| !part.is_empty() && *part != ".." && !part.contains('\\');
    let mut file = root.to_path_buf();
    path.split('/')
        .filter(safe)
        .for_each(|part| file.push(part));
    if file.is_dir() && !path.ends_with('/') {
        let head = format!("HTTP/1.1 301 Moved Permanently\r\nLocation: {path}/\r\n");
        return stream
            .write_all(format!("{head}Content-Length: 0\r\nConnection: close\r\n\r\n").as_bytes());
    }
    if file.is_dir() {
        file.push("index.html");
    }
    let (status, body) = match std::fs::read(&file) {
        Ok(body) => ("200 OK", body),
        Err(_) => ("404 Not Found", b"not found".to_vec()),
    };
    let kind = match file.extension().and_then(|e| e.to_str()) {
        Some("html") => "text/html; charset=utf-8",
        Some("js") => "text/javascript",
        Some("wasm") => "application/wasm",
        _ => "application/octet-stream",
    };
    let head = format!(
        "HTTP/1.1 {status}\r\nContent-Type: {kind}\r\nContent-Length: {}\r\n",
        body.len()
    );
    stream.write_all(format!("{head}Connection: close\r\n\r\n").as_bytes())?;
    stream.write_all(&body)
}
