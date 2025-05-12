#include "server.hpp"
#include <Geode/Geode.hpp>
#include <thread>
#include <string>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <unistd.h>
#include "controls.hpp"

using namespace geode::prelude;

namespace tcpserver
{
    int frameSocket = -1;

    void sendFrame(unsigned char *buffer, int width, int height)
    {
        if (frameSocket < 0)
            return;

        int meta[2] = {width, height};
        send(frameSocket, meta, sizeof(meta), 0);

        int imageSize = width * height * 4;
        send(frameSocket, buffer, imageSize, 0);
    }

    void serverThreadFunction()
    {
        int server_fd, new_socket;
        struct sockaddr_in address;
        int opt = 1;
        int addrlen = sizeof(address);

        server_fd = socket(AF_INET, SOCK_STREAM, 0);
        setsockopt(server_fd, SOL_SOCKET, SO_REUSEADDR | SO_REUSEPORT, &opt, sizeof(opt));

        address.sin_family = AF_INET;
        address.sin_addr.s_addr = INADDR_ANY;
        address.sin_port = htons(22222);

        bind(server_fd, (struct sockaddr *)&address, sizeof(address));
        listen(server_fd, 3);

        log::info("Command server listening on port {}", ntohs(address.sin_port));

        while (true)
        {
            new_socket = accept(server_fd, (struct sockaddr *)&address, (socklen_t *)&addrlen);
            char buffer[1024] = {0};
            read(new_socket, buffer, 1024);

            std::string command(buffer);
            log::info("Received command: {}", command);

            if (command.find("reset") != std::string::npos)
            {
                geode::queueInMainThread([]
                                         { controls::resetLevel(); });
            }

            if (command.find("step") != std::string::npos)
            {
                bool press = command.find("jump") != std::string::npos || command.find("hold") != std::string::npos;
                controls::step(5, press);
            }

            // TODO: Ishan can you make this send info like level %, whether we died this frame, etc?
            // Do it in JSON or some other format, up to you. We just need to parse it once we receive in Python
            const char *response = "ok";
            send(new_socket, response, strlen(response), 0);
        }
    }

    void frameReceiverThread()
    {
        int server_fd;
        struct sockaddr_in address;
        int opt = 1;
        int addrlen = sizeof(address);

        server_fd = socket(AF_INET, SOCK_STREAM, 0);
        setsockopt(server_fd, SOL_SOCKET, SO_REUSEADDR | SO_REUSEPORT, &opt, sizeof(opt));

        address.sin_family = AF_INET;
        address.sin_addr.s_addr = INADDR_ANY;
        address.sin_port = htons(22223);

        bind(server_fd, (struct sockaddr *)&address, sizeof(address));
        listen(server_fd, 1);

        log::info("Frame socket listening on port 22223...");

        frameSocket = accept(server_fd, (struct sockaddr *)&address, (socklen_t *)&addrlen);
        log::info("Frame client connected.");
    }

    void start()
    {
        std::thread(serverThreadFunction).detach();
        std::thread(frameReceiverThread).detach();
    }
} // namespace tcpserver
