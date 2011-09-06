
/**
 * @file  v4l.c
 * @brief Implemenatation for interacting with cameras using V4L2
 */

#include <stdio.h>
#include <stdlib.h>
#include <sys/fcntl.h>
#include <sys/ioctl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <string.h>
#include <assert.h>
#include <sys/time.h> /* for videodev2.h */
#include <linux/videodev2.h>

#include "v4l.h"

#define CLEAR(x) memset(&(x), 0, sizeof((x)))



/**
 * @brief opens a camera device and returns its file descriptor
 *
 * @param filename  the file name (e.g. '/dev/video0') of the camera
 * @return          the file dscriptor on successful opening, -1 otherwise
 */
int koki_v4l_open_cam(const char* filename)
{

	int fd;

	assert(filename != NULL);

	fd = open(filename, O_RDWR, 0);
	if (fd == -1){
		fprintf(stderr, "couldn't open cam\n");
	}

	return fd;

}



/**
 * @brief closes the camera
 *
 * @param fd  the camera's open file descriptor
 */
void koki_v4l_close_cam(int fd)
{

	close(fd);

}



/**
 * @brief gets the camera's current format (size, colour format, etc...)
 *
 * @param fd  the camera's file descriptor
 * @return    a filled V4L2's format structure
 */
struct v4l2_format koki_v4l_get_format(int fd)
{

	struct v4l2_format fmt;
	int ret;

	CLEAR(fmt);

	fmt.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
	ret = ioctl(fd, VIDIOC_G_FMT, &fmt);

	assert(ret >= 0);

	return fmt;

}



/**
 * @brief outputs to \c stdout a format structure
 *
 * @param fmt  the V4L2 format structure
 */
void koki_v4l_print_format(struct v4l2_format fmt)
{
	printf("Camera Format:\n");

	printf("  bufer type: %d\n", fmt.type);

	printf("  dimensions: %d x %d\n",
	       fmt.fmt.pix.width, fmt.fmt.pix.height);

#define pix_fmt fmt.fmt.pix.pixelformat
	printf("  colour format: %c%c%c%c\n",
	        pix_fmt        & 0xFF,
	       (pix_fmt >> 8)  & 0xFF,
	       (pix_fmt >> 16) & 0xFF,
	       (pix_fmt >> 24) & 0xFF);
#undef pix_fmt

	printf("  field: %d\n", fmt.fmt.pix.field);

	printf("  bytes per line: %d\n", fmt.fmt.pix.bytesperline);

}



/**
 * @brief returns a YUYV format structure ready to send to the camera
 *
 * @param w  the desired width of the image
 * @param h  the desired height of the image
 * @return   a YUYV V4L2 format structure
 */
struct v4l2_format koki_v4l_create_YUYV_format(unsigned int w, unsigned int h)
{

	struct v4l2_format fmt;

	CLEAR(fmt);

	fmt.type                 = V4L2_BUF_TYPE_VIDEO_CAPTURE;
	fmt.fmt.pix.width        = w;
	fmt.fmt.pix.height       = h;
	fmt.fmt.pix.pixelformat  = V4L2_PIX_FMT_YUYV;
	fmt.fmt.pix.bytesperline = 2 * w;
	fmt.fmt.pix.sizeimage    = 2 * w * h;

	return fmt;

}



/**
 * @brief tries to set the format of the camera to the format specified
 *
 * @param fd  the camera's file descriptor
 * @param fmt the format to set
 * @return    a negative value on failure
 */
int koki_v4l_set_format(int fd, struct v4l2_format fmt)
{

	int ret;

	ret = ioctl(fd, VIDIOC_S_FMT, &fmt);
	if (ret < 0){
		fprintf(stderr, "unable to set format\n");
	}

	return ret;

}



/**
 * @brief returns the camera's capabilities (e.g. driver, streaming, etc...)
 *
 * @param fd  the camera's file descriptor
 * @return    a filled V4L2 capabilities structure for the camera
 */
struct v4l2_capability koki_v4l_get_capability(int fd)
{

	struct v4l2_capability cap;
	int ret;

	CLEAR(cap);

	ret = ioctl(fd, VIDIOC_QUERYCAP, &cap);
	if (ret < 0){
		fprintf(stderr, "couldn't get capabilities\n");
	}

	return cap;

}



/**
 * @brief prints to \c stdout a V4L2 capabilities structure
 *
 * @param cap  a V4L2 capability structure to output
 */
void koki_v4l_print_capability(struct v4l2_capability cap)
{

	printf("Capability:\n");
	printf("  driver: %s\n", cap.driver);
	printf("  card: %s\n", cap.card);
	printf("  version: %X\n", cap.version);
	printf("  capabilities: %X\n", cap.capabilities);
	printf("   - streaming?  %s\n",
	       cap.capabilities & V4L2_CAP_STREAMING ? "YES" : "NO");

}



/**
 * @brief returns the control value for the specified V4L2 control ID
 *
 * @param fd  the camera's file descriptor
 * @param id  the V4L2 control ID
 * @return    the value associated with the given control value
 */
int koki_v4l_get_control(int fd, unsigned int id)
{

	struct v4l2_control ctrl;
	int ret;

	CLEAR(ctrl);

	ctrl.id = id;

	ret = ioctl(fd, VIDIOC_G_CTRL, &ctrl);
	if (ret < 0){
		fprintf(stderr, "unable to get control\n");
	}

	return ctrl.value;

}


/**
 * @brief sets the V4L2 control ID as the given value
 *
 * @param fd     the camera's file descriptor
 * @param id     the V4L2 control ID to modify
 * @param value  the new value to set the control to
 */
int koki_v4l_set_control(int fd, unsigned int id, unsigned int value)
{

	struct v4l2_control ctrl;
	int ret;

	CLEAR(ctrl);

	ctrl.id = id;
	ctrl.value = value;

	ret = ioctl(fd, VIDIOC_S_CTRL, &ctrl);
	if (ret < 0){
		fprintf(stderr, "unable to set control\n");
	}

	return ret;

}



/**
 * @brief allocates all of the buffers required for memory mapped IO with
 *        the camera
 *
 * @param fd     the camera's file descriptor
 * @param count  the number of buffers to allocate (and where to store the
 *               number of actually allocated buffers)
 * @return       an array of \c koki_buffer_ts of length \c count (note that
 *               \c count may have changed after this function returns)
 */
koki_buffer_t* koki_v4l_prepare_buffers(int fd, int *count)
{

	struct v4l2_requestbuffers reqbuf;
	struct v4l2_buffer buffer;
	int ret;
	koki_buffer_t *buffers;

	CLEAR(reqbuf);

	reqbuf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
	reqbuf.memory = V4L2_MEMORY_MMAP;
	reqbuf.count = *count;

	ret = ioctl(fd, VIDIOC_REQBUFS, &reqbuf);

	if (ret < 0){
		fprintf(stderr, "couldn't request buffers\n");
		*count = 0;
		return NULL;
	}

	*count = reqbuf.count;

	buffers = calloc(*count, sizeof(koki_buffer_t));
	if (buffers == NULL){
		fprintf(stderr, "couldn't allocate buffer pointers\n");
		return NULL;
	}

	for (int i=0; i < *count; i++){

		CLEAR(buffer);

		buffer.type = reqbuf.type;
		buffer.memory = V4L2_MEMORY_MMAP;
		buffer.index = i;

		ret = ioctl(fd, VIDIOC_QUERYBUF, &buffer);
		if (ret < 0){
			fprintf(stderr, "querybuf request failure\n");
			free(buffers);
			return NULL;
		}

		buffers[i].length = buffer.length;
		buffers[i].start = mmap(NULL, buffer.length,
					PROT_READ | PROT_WRITE,
					MAP_SHARED, fd,
					buffer.m.offset);

		if (buffers[i].start == MAP_FAILED){
			fprintf(stderr, "map failed\n");
			free(buffers);
			return NULL;
		}

		memset(buffers[i].start, 0x41, buffers[i].length);

	}

	return buffers;

}



/**
 * @brief tells the camera to start streaming so that frames can be grabbed
 *
 * @param fd  the camera's file descriptor
 * @return    a negative value on failure
 */
int koki_v4l_start_stream(int fd)
{

	int type, ret;

	type = V4L2_BUF_TYPE_VIDEO_CAPTURE;

	ret = ioctl(fd, VIDIOC_STREAMON, &type);
	if (ret < 0){
		fprintf(stderr, "failed to start stream\n");
	}

	return ret;

}



/**
 * @brief tells the camera to stop stremaing
 *
 * This shoud be done before the camera is closed.
 *
 * @param fd  the camera's file descriptor
 * @return    a negative number on failure
 */
int koki_v4l_stop_stream(int fd)
{

	int type, ret;

	type = V4L2_BUF_TYPE_VIDEO_CAPTURE;

	ret = ioctl(fd, VIDIOC_STREAMOFF, &type);
	if (ret < 0){
		fprintf(stderr, "failed to stop stream\n");
	}

	return ret;

}


