import yt_dlp  # type: ignore

class Download:
    def __init__(self, url):
        self.url = url
        self.ydl_opts = {}

    def configs(self):
        self.ydl_opts = {
            "format": "bestaudio/best",
            "outtmpl": r"E:\Projects\Github\Extract-Data-from-youtube\%(title)s.%(ext)s",
            "ffmpeg_location": r"E:\Projects\Github\Extract-Data-from-youtube\ffmpeg.exe",
            "cookiefile": r"E:\Projects\Github\Extract-Data-from-youtube\www.youtube.com_cookies.txt",
            "writesubtitles": True,          
            "writeautomaticsub": True,       
            "subtitleslangs": ["fa"], 
            "subtitlesformat": "srt",       
            "postprocessors": [{
                "key": "FFmpegExtractAudio",
                "preferredcodec": "mp3",
                "preferredquality": "192",
            }],
        }

    def download(self):
        if not self.ydl_opts:
            self.configs()  
        with yt_dlp.YoutubeDL(self.ydl_opts) as ydl:
            ydl.download([self.url])


video = Download("https://www.youtube.com/watch?v=uOXmCM9ZWo0") # Your Random Url for Extract 
video.download()
