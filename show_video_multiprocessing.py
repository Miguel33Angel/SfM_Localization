from multiprocessing import Pool, Queue
import time
import cv2


# intialize global variables for the pool processes:
def init_pool(d_b):
    global detection_buffer
    detection_buffer = d_b


def detect_object(frame):
    time.sleep(1)
    detection_buffer.put(frame)


def show():
    while True:
        frame = detection_buffer.get()
        if frame is not None:
            cv2.imshow("Video", frame)
            cv2.waitKey(180)
        else:
            break
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    return


# required for Windows:
if __name__ == "__main__":

    detection_buffer = Queue()
    # 6 workers: 1 for the show task and 5 to process frames:
    pool = Pool(6, initializer=init_pool, initargs=(detection_buffer,))
    # run the "show" task:
    show_future = pool.apply_async(show)

    video_name = "secuencia_a_cam2.avi"
    #vs = VideoStreamer(video_name, [640, 480], 1, ['*.png', '*.jpg', '*.jpeg'], 1000000)
    cap = cv2.VideoCapture(video_name)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 4)
    # w, h = 640, 480

    # Get frame_
    i_frame = 0
    cap.set(cv2.CAP_PROP_POS_FRAMES, i_frame)
    ret, ref_frame = cap.read()
    # ref_frame = cv2.resize(ref_frame, (w, h), cv2.INTER_AREA)
    # ref_frame = cv2.cvtColor(ref_frame, cv2.COLOR_RGB2GRAY)
    i_frame = i_frame+1
    # ret ,ref_frame = vs.next_frame()


    #cap = cv2.VideoCapture(0)

    futures = []
    while True:
        cap.set(cv2.CAP_PROP_POS_FRAMES, i_frame)
        ret, frame = cap.read()
        if not ret:
            break

        # frame = cv2.resize(frame, (w, h), cv2.INTER_AREA)
        # frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        i_frame = i_frame + 5
        f = pool.apply_async(detect_object, args=(frame,))
        futures.append(f)
        time.sleep(0.18)
    # wait for all the frame-putting tasks to complete:
    for f in futures:
        f.get()
    # signal the "show" task to end by placing None in the queue
    detection_buffer.put(None)
    show_future.get()
    print("program ended")