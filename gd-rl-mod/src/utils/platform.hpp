#pragma once

#if defined(_WIN32)
// #undef UNICODE
// #define WIN32_LEAN_AND_MEAN

// #include <windows.h>
// #include <winsock2.h>
// #include <ws2tcpip.h>
#include <GL/gl.h>
// #pragma comment(lib, "Ws2_32.lib")
using socklen_t = int;

#define CLOSE_SOCKET closesocket
#define SOCKET_READ(s, b, l, f) recv(s, b, l, f)

#else
#include <OpenGL/gl.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <sys/socket.h>
#include <unistd.h>

#define CLOSE_SOCKET close
#define SOCKET_READ(s, b, l, f) read(s, b, l)
#endif
