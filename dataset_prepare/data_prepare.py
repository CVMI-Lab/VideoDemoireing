"""
This file contains functions that we used to prepare our video demoireing data.
"""

import cv2, glob, os, argparse
import numpy as np

def crop_scale(t_h, t_w, image):
    """
    Args:
        t_h: target height
        t_w: target width
        image: original image, BGR, HxWxC

    Returns: image after cropping and scaling
    """
    h, w, _ = image.shape
    s_w = t_w/w
    s_h = t_h/h
    if s_w > s_h:
        image_new = cv2.resize(image, (int(w*s_w), int(h*s_w)))
        image_new = image_new[0:t_h, 0:t_w, :]
    else:
        image_new = cv2.resize(image, (int(w*s_h), int(h*s_h)))
        image_new = image_new[0:t_h, 0:t_w, :]
    return image_new


def video_read(video_path, output_path):
    """
    Args:
        video_path: video path
        output_path: path used to store video frames

    Returns:
    """
    vidcap = cv2.VideoCapture(video_path)
    success, image = vidcap.read()
    count = 0
    while success:
        cv2.imwrite("%s/%05d.jpg" % (output_path, count), image)  # save frame
        success, image = vidcap.read()
        count += 1
    vidcap.release()


def video_write(frames, output_path, add_flag=False, flag_img=None, num_flag=15, fps=30):
    """
    Args:
        frames: frames used to write the video, a list
        output_path: video output path
        add_flag: If Ture, add auxiliary frames at the beginning/end of the video
        flag_img: the flag image, pure white here, same size with frame HxWxC
        num_flag: number of flag images added to video
        fps: frame rate

    Returns:
    """
    if add_flag:
        flags = []
        for i in range(num_flag):
            flags.append(flag_img)
        frames = flags + frames + flags
    num_frames = len(frames)
    print('num_frames: %s' % num_frames)
    h, w, _ = frames[0].shape
    size = (w, h)
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'DIVX'), fps, size)  # 'DIVX' for .avi video
    for i in range(num_frames):
        out.write(frames[i])
    out.release()
    
    
def delete_outlier(points1, points2, move=0, outlier_percent=0.3):
    """
    Delete the outliers/mismatches based on the angle and distance.
    Args:
        points1: key points detected in frame1
        points2: key points detected in frame2
        move: move for a small distance to avoid points appear at the same location
        outlier_percent: how many outliers are removed

    Returns: indexes of selected key points
    """
    # angle
    points1_mv = points1.copy()
    points1_mv[:, 0] = points1[:, 0] - move
    vecs = points2 - points1_mv
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    vec_norms = vecs/(norms+1e-6)
    vec_means = np.mean(vec_norms, axis=0).reshape((2, 1))
    cross_angles = vec_norms.dot(vec_means)[:, 0]
    index = np.argsort(-cross_angles)
    num_select = int(len(index) * (1-outlier_percent))
    index_selected = index[0:num_select]
    
    # distance
    index1 = np.argsort(norms[:, 0])
    index1_selected = index1[0:num_select]
    
    index_selected = list(set(index1_selected) & set(index_selected))

    return index_selected


def ecc_align(im1, im2):
    """
    Align images with ecc algorithm
    Args:
        im1: the reference image, HxWxC
        im2: image need to be warped, HxW,C

    Returns: warped image and warping matrix
    """
    # Convert images to grayscale
    im1_gray = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
    im2_gray = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)

    # Find size of image1
    sz = im1.shape

    # Define the motion model
    warp_mode = cv2.MOTION_HOMOGRAPHY

    # Define 2x3 or 3x3 matrices and initialize the matrix to identity
    if warp_mode == cv2.MOTION_HOMOGRAPHY:
        warp_matrix = np.eye(3, 3, dtype=np.float32)
    else:
        warp_matrix = np.eye(2, 3, dtype=np.float32)

    # Specify the number of iterations.
    number_of_iterations = 1000

    # Specify the threshold of the increment
    # in the correlation coefficient between two iterations
    termination_eps = 1e-10

    # Define termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, number_of_iterations, termination_eps)

    # Run the ECC algorithm. The results are stored in warp_matrix.
    # mask = None
    mask = np.ones_like(im2_gray)
    (cc, warp_matrix) = cv2.findTransformECC(im2_gray, im1_gray, warp_matrix, warp_mode, criteria, mask, 5)

    if warp_mode == cv2.MOTION_HOMOGRAPHY:
        # Use warpPerspective for Homography
        # im2_aligned = cv2.warpPerspective(im2, warp_matrix, (sz[1], sz[0]), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
        im2_aligned = cv2.warpPerspective(im2, warp_matrix, (sz[1], sz[0]))
    else:
        # Use warpAffine for Translation, Euclidean and Affine
        # im2_aligned = cv2.warpAffine(im2, warp_matrix, (sz[1], sz[0]), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
        im2_aligned = cv2.warpAffine(im2, warp_matrix, (sz[1], sz[0]))

    # Show final results
    # cv2.imshow("Image 1", im1)
    # cv2.imshow("Image 2", im2)
    # cv2.imshow("Aligned Image 2", im2_aligned)
    # cv2.waitKey(0)
    return im2_aligned, warp_matrix


def align_homography(img1, img2, save_match=False, max_features=600, good_match_percent=0.8, outlier=False, method='orb', use_black_boundary=True):
    """
    Align images using homography
    Args:
        img1: BGR image1, HxWxC, warp
        img2: BGR image2, HxWxC, reference
        save_match: if True, draw matches between two images
        max_features: maximum number of detected key points
        good_match_percent: percentage of good match
        outlier: if True, delete the outliers based on angle and distance
        method: the method used to detect key points, currently, only support 'orb'
        use_black_boundary: if True, use detected points in black regions

    Returns: aligned images and homography matrix
    Note: for different frames, the optimal parameters (e.g., max_features) will be different. You can adjust them.
    """

    im1Gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    im2Gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    
    if not use_black_boundary:
        # Transform black regions to white background
        im2Gray[np.where(im2Gray < 5)] = 255

    if method == 'orb':
        # Detect ORB features and compute descriptors.
        orb = cv2.ORB_create(max_features)
        keypoints1, descriptors1 = orb.detectAndCompute(im1Gray, None)
        keypoints2, descriptors2 = orb.detectAndCompute(im2Gray, None)
    else:
        pass
        import pdb; pdb.set_trace()
        # # Detect SIFT features and compute descriptors.
        # sift = cv2.SIFT_create(max_features)
        # keypoints1, descriptors1 = sift.detectAndCompute(im1Gray, None)
        # keypoints2, descriptors2 = sift.detectAndCompute(im2Gray, None)

    # Match features.
    matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
    matches = matcher.match(descriptors1, descriptors2, None)

    # Sort matches by score
    matches.sort(key=lambda x: x.distance, reverse=False)

    # Remove not so good matches
    numGoodMatches = int(len(matches) * good_match_percent)
    matches = matches[:numGoodMatches]

    # Extract location of good matches
    points1 = np.zeros((len(matches), 2), dtype=np.float32)
    points2 = np.zeros((len(matches), 2), dtype=np.float32)

    for i, match in enumerate(matches):
        points1[i, :] = keypoints1[match.queryIdx].pt
        points2[i, :] = keypoints2[match.trainIdx].pt
    
    # Delete outliers
    if outlier:
        index = delete_outlier(points1, points2)
        matches = list(np.array(matches)[index])
        points1 = points1[index, :]
        points2 = points2[index, :]

    if save_match:
        # Draw top matches
        imMatches = cv2.drawMatches(img1, keypoints1, img2, keypoints2, matches, None)
        cv2.imwrite("matches.jpg", imMatches)

    # Find homography
    homo, mask = cv2.findHomography(points1, points2, cv2.RANSAC, 7)

    # Use homography
    height, width, channels = img2.shape
    im1Align = cv2.warpPerspective(img1, homo, (width, height))

    return im1Align, homo


def boundary(img):
    """
    Add auxiliary black regions
    Args:
        img: BGR image, HxWxC

    Returns: image with black regions
    """
    h, w, c = img.shape
    h_new = int(h*1.2)
    w_new = int(w*1.2)
    img_new = np.ones((h_new, w_new, c))*255.0
    img_new[int(0+0.075*h):int(h_new-0.075*h), int(0+0.075*w):int(w_new-0.075*w), :] = 0
    img_new[int(0.05*h):int(h_new-0.05*h), int(0.25*w_new):int(0.75*w_new), :] = 0
    img_new[int(0.25*h_new):int(0.75*h_new), int(0.05*w):int(w_new-0.05*w), :] = 0
    img_new[int(0.1*h):int(0.1*h+h), int(0.1*w):int(0.1*w+w), :] = img

    return np.uint8(img_new)


def main():
    """
     This function is used to pre-process the original video frames, including flag images, auxiliary black areas,
     frame rates, frame size adjustments.
    """
    def add_args():
        parser = argparse.ArgumentParser()
        parser.add_argument("--target_w", type=int, default=1280, help='the width of frame')
        parser.add_argument("--target_h", type=int, default=720, help='the height of frame')
        parser.add_argument("--add_boundary", type=str, default=True, help='add auxiliary black areas')
        parser.add_argument("--add_flag", type=str, default=True, help='extend video with auxiliary frames')
        parser.add_argument("--num_flags", type=int, default=15, help='number of auxiliary frames')
        parser.add_argument("--fps", type=int, default=10, help='the frame rate of video')
        parser.add_argument("--scene_type", type=str, default='landscape', help='animal, daily, house, landscape, Moca, sports, text, Reds')
        parser.add_argument("seq_path", type=str, default='./original/frames/', help='path of original video frames')
        parser.add_argument("output_frames", type=str, default='./processed/frames/', help='path to store pre-processed frames')
        parser.add_argument("output_videos", type=str, default='./processed/videos/', help='path to store pre-processed videos')
        return parser.parse_args()

    args = add_args()
    flag_image = np.ones((args.target_h, args.target_w, 3))*255
    cnt = 0
    seq_path = args.seq_path + args.scene_type + '/'
    output_frames = args.output_frames + args.scene_type + '/'
    output_videos = args.output_videos + args.scene_type + '/'
    if args.add_boundary:
        flag_image = boundary(flag_image)

    if not os.path.isdir(output_frames):
        os.makedirs(output_frames)
    if not os.path.isdir(output_videos):
        os.makedirs(output_videos)
    seq_names = sorted(glob.glob(seq_path + '*/'))

    for j in range(len(seq_names)):
        print('There are %s scenes in %s.' % (j, args.scene_type))
        seq_name = seq_names[j]
        image_names = sorted(glob.glob(seq_name + '*.jpg')) + sorted(glob.glob(seq_name + '*.png'))
        frames = []
        if len(image_names) < 60:  # default at least 60 frames each video
            continue

        if not os.path.isdir(output_frames + 'video_%s/' % j):
            os.makedirs(output_frames + 'video_%s/' % j)

        for i in range(60):
            image = cv2.imread(image_names[i])
            image_resized = crop_scale(t_h=args.target_h, t_w=args.target_w, image=image)
            if args.add_boundary:
                image_resized = boundary(image_resized)
            frames.append(image_resized)
            cv2.imwrite(output_frames + 'video_%s/%05d.jpg' % (j, i+args.num_flags), image_resized)
            cnt = cnt + 1
        print('Begin video write:')
        video_write(frames=frames, output_path=output_videos + 'video_%s.avi' % j,
                    add_flag=args.add_flag, flag_img=flag_image, num_flag=args.num_flags, fps=args.fps)


def main1():
    """
    Extract frames from moire videos.
    """
    def add_args():
        parser = argparse.ArgumentParser()
        parser.add_argument("--moire_video_path", type=str, default='./moire/tcl/videos/', help='path of moire videos')
        parser.add_argument("--output_frames", type=str, default='./moire/tcl/frames/', help='path to store moire frames')
        parser.add_argument("--video_type", type=str, default='*.avi', help='the type of video')
        return parser.parse_args()

    args = add_args()
    folders = ['Moca', 'landscape', 'sports', 'daily', 'house', 'text', 'Reds', 'animal']
    
    for i in range(len(folders)):
        folder = folders[i]
        moire_video_path = args.moire_video_path + '%s/' % folder
        output_root = args.output_frames + '%s/' % folder
        video_names = sorted(glob.glob(moire_video_path + args.video_type))
        
        for j in range(len(video_names)):
            video_name = video_names[j]
            print(video_name)
            output_path = output_root + video_name.split('/')[-1].split('.')[0]
            if not os.path.isdir(output_path):
                os.makedirs(output_path)
            video_read(video_path=video_name, output_path=output_path)


def main2():
    """
    Align the camera captured frames to reference frames.
    """
    def add_args():
        parser = argparse.ArgumentParser()
        parser.add_argument('--frame_path1', type=str, default='./moire/tcl/frames/train/', help='moire frames path')
        parser.add_argument('--frame_path2', type=str, default='./processed/frames/train/', help='clean frames path')
        parser.add_argument('--output_path', type=str, default='./homo_align/tcl/frames/train/', help='path to store aligned frames')
        parser.add_argument('--begin_position', type=int, default=0, help='index of the first clean frame')
        return parser.parse_args()

    args = add_args()
    folders = ['Reds', 'Moca', 'landscape', 'sports', 'daily', 'house', 'text', 'animal']

    for ii in range(len(folders)): 
        folder = folders[ii]
        frame_paths1 = sorted(glob.glob(args.frame_path1 + '%s/' % folder + '*/'))
        frame_paths2 = sorted(glob.glob(args.frame_path2 + '%s/' % folder + '*/'))
        assert len(frame_paths1) == len(frame_paths2)
        for jj in range(len(frame_paths1)):
            frame_path1 = frame_paths1[jj]
            frame_path2 = frame_paths2[jj]
            video_index1 = frame_path1.split('/')[-1]
            video_index2 = frame_path2.split('/')[-1]
            assert video_index1 == video_index2
            output_path = args.output_path + '%s/%s/' % (folder, video_index1)
            
            # # manually define the path
            # frame_path1 = './moire/tcl/frames/val/Reds/video_6/'
            # frame_path2 = './processed/frames/val/Reds/video_6/'
            # output_path = './homo_align/tcl/video_6/'
            
            if not os.path.isdir(output_path):
                os.makedirs(output_path)
        
            names_1 = sorted(glob.glob(frame_path1 + '*.jpg'))
            names_2 = sorted(glob.glob(frame_path2 + '*.jpg'))
            num1 = len(names_1)
            num2 = len(names_2)
            step = float(num1)/float(60)
        
            for i in range(60):
                # import pdb; pdb.set_trace()
                # select the medium one.
                index = int(step/2 + i*step)

                # Read reference clean image
                refFilename = names_2[i + args.begin_position]
                print("Reading reference clean image: ", refFilename)
                imReference = cv2.imread(refFilename, cv2.IMREAD_COLOR)
                 
#                # manually adjustments (choose ROI regions)
#                imReference_new = np.ones_like(imReference)*100
#                imReference_new[200:700, 300:1000, :] = imReference[200:700, 300:1000, :]
#                imReference = imReference_new

                # Read moire images
                imFilename = names_1[index] 
                print("Reading moire image to align: ", imFilename)
                img = cv2.imread(imFilename, cv2.IMREAD_COLOR)

                # begin align
                print("Aligning images ...")
                imAlign, homo = align_homography(img, imReference, save_match=True, max_features=400, good_match_percent=0.6, outlier=True, use_black_boundary=True)
                # imAlign, homo = align_homography(imReference, im, save_match=True, outlier=True)  # align GT to moire

                # Write aligned image to disk.
                outFilename = output_path + "%05d.jpg" % i
                print("Saving aligned image: ", outFilename)
                cv2.imwrite(outFilename, imAlign)

def main3():
    """
    Generate training/testing pairs.
    """

    def add_args():
        parser = argparse.ArgumentParser()
        parser.add_argument('--target_path', type=str, default='./processed/frames/train/', help='path of clean images')
        parser.add_argument('--source_path', type=str, default='./homo_align/tcl/frames/train/', help='path of aligned moire images')
        parser.add_argument('--output_target', type=str, default='../dataset/tcl/target/train/', help='path of training target')
        parser.add_argument('--output_source', type=str, default='../dataset/tcl/source/train/', help='path of training source')
        parser.add_argument('--begin_position', type=int, default=15, help='index of the first clean frame')
        return parser.parse_args()

    args = add_args()
    folders = ['Reds', 'Moca', 'landscape', 'sports', 'daily', 'house', 'text', 'animal']
    # folders = ['Reds']

    if not os.path.isdir(args.output_source):
        os.makedirs(args.output_source)

    if not os.path.isdir(args.output_target):
        os.makedirs(args.output_target)

    for k in range(len(folders)):
        folder = folders[k]
        source_paths = sorted(glob.glob(args.source_path + '%s/' % folder + '*/'))
        target_paths = sorted(glob.glob(args.target_path + '%s/' % folder + '*/'))
        print('%s: %s' % (folder, len(source_paths)))
        assert len(source_paths) == len(target_paths)

        # if not folder == 'animal':
        #     continue
        # for i in range(130):
        #     print(source_paths[i])
        #     print(target_paths[i])
        #     continue

        for i in range(len(source_paths)):
            source_path = source_paths[i]
            target_path = target_paths[i]
            video_index_source = int(source_path.split('/')[-2].split('_')[-1])
            video_index_target = int(target_path.split('/')[-2].split('_')[-1])
            assert video_index_source == video_index_target

            for j in range(60):
                src_img = cv2.imread(source_path + '/%05d.jpg' % j)
                tar_img = cv2.imread(target_path + '/%05d.jpg' % (j + args.begin_position))
                h, w, c = tar_img.shape
                # crop 720x1280
                src_img_center = src_img[int(h / 2 - 360):int(h / 2 + 360), int(w / 2 - 640):int(w / 2 + 640)]
                tar_img_center = tar_img[int(h / 2 - 360):int(h / 2 + 360), int(w / 2 - 640):int(w / 2 + 640)]
                if folder == 'Reds':
                    cv2.imwrite(args.output_source + 'v1%s%03d_%05d.jpg' % (k, video_index_source, j), src_img_center)
                    cv2.imwrite(args.output_target + 'v1%s%03d_%05d.jpg' % (k, video_index_target, j), tar_img_center)
                else:
                    cv2.imwrite(args.output_source + 'v2%s%03d_%05d.jpg' % (k-1, video_index_source, j), src_img_center)
                    cv2.imwrite(args.output_target + 'v2%s%03d_%05d.jpg' % (k-1, video_index_target, j), tar_img_center)


if __name__=='__main__':
    # version 1, run step by step
    # # step1: prepare videos for moire video capture
    # main()

    # # step2: extract frames from moire videos
    # main1()

    # step3: manually remove auxiliary white frames

    # # step4: align frames using homography (Here, we align moire frames to GT. You can also align GT to moire frames.)
    # main2()

    # step5: center cropped 720x1280 frames for training/testing pairs
    main3()

    # # Toy example of alignment
    # img1 = cv2.imread('2.jpg')
    # img2 = cv2.imread('1.jpg')
    # imReg, h = align_homography(img1, img2, save_match=True)
    # cv2.imwrite('align.png', imReg)

    # version 2, use the estimated optical flow to further refine the aligned frame.
    # Not included here. please refer to the RAFT to align two frames with the estimated optical flow.



