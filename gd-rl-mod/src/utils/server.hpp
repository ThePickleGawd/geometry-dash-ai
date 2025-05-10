#pragma once

namespace tcpserver
{
    void start();
    void sendFrame(unsigned char *buffer, int width, int height);
}
