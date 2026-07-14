import gdown

file_id = "1dX1bCthvy_9Q_qRS5Qf_17zlsaziYs61"

# The use_cookies parameter is True by default, and gdown looks at ~/.cache/gdown/cookies.txt
gdown.download(id=file_id, use_cookies=True, quiet=False)
