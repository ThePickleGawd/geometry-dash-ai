#include "server.hpp"
#include <Geode/Geode.hpp>
#include <Geode/modify/PlayLayer.hpp>
#include <Geode/modify/AppDelegate.hpp>
#include <thread>
#include <string>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <string>
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
            log::info("Command client connected.");

            char buffer[1024];

            while (true)
            {
                memset(buffer, 0, sizeof(buffer));
                int bytesRead = read(new_socket, buffer, sizeof(buffer));

                if (bytesRead <= 0)
                {
                    log::info("Client disconnected or error.");
                    close(new_socket);
                    break;
                }

                std::string command(buffer);
                log::info("Received command: {}", command);

                PlayLayer *pl = PlayLayer::get();
                if (!pl || !pl->m_player1)
                {
                    std::string errorResponse = R"({"error": "Game not running or not initialized"})";
                    send(new_socket, errorResponse.c_str(), errorResponse.size(), 0);
                    continue;
                }

                if (command.find("reset") != std::string::npos)
                {
                    int percent = 1; // default
                    std::istringstream iss(command);
                    std::string word;
                    while (iss >> word)
                    {
                        try
                        {
                            int val = std::stoi(word);
                            if (val >= 1 && val <= 99)
                            {
                                percent = val;
                                break;
                            }
                        }
                        catch (...)
                        {
                            // skip non-integer words
                        }
                    }

                    geode::queueInMainThread([percent]
                                             {
                        controls::resetLevel(); 
                        controls::loadFromPercent(percent); });
                }

                if (command.find("step") != std::string::npos)
                {
                    bool press = command.find("jump") != std::string::npos || command.find("hold") != std::string::npos;
                    controls::step(4, press);
                }

                bool died = pl->m_player1->m_isDead;
                float percent = (pl->m_player1->getPositionX() / pl->m_levelLength) * 100.0f;
                std::string response = fmt::format(R"({{"dead": {}, "percent": {}}})", died ? "true" : "false", percent);

                log::info("Sending response: {}", response);
                send(new_socket, response.c_str(), response.size(), 0);
            }
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
