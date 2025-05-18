#include "platform.hpp"

#include "server.hpp"
#include <Geode/Geode.hpp>
#include <Geode/modify/PlayLayer.hpp>
#include <Geode/modify/AppDelegate.hpp>
#include <thread>
#include <string>
#include <sstream>
#include <cstring>
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
        send(frameSocket, reinterpret_cast<const char *>(meta), sizeof(meta), 0);

        int imageSize = width * height * 4;
        send(frameSocket, reinterpret_cast<const char *>(buffer), imageSize, 0);
    }

    void serverThreadFunction()
    {
        int server_fd, new_socket;
        struct sockaddr_in address;
        int opt = 1;
        socklen_t addrlen = sizeof(address);

        server_fd = socket(AF_INET, SOCK_STREAM, 0);
#if defined(_WIN32)
        setsockopt(server_fd, SOL_SOCKET, SO_REUSEADDR, reinterpret_cast<const char *>(&opt), sizeof(opt));
#else
        setsockopt(server_fd, SOL_SOCKET, SO_REUSEADDR | SO_REUSEPORT, &opt, sizeof(opt));
#endif

        address.sin_family = AF_INET;
        address.sin_addr.s_addr = INADDR_ANY;
        address.sin_port = htons(22222);

        bind(server_fd, reinterpret_cast<struct sockaddr *>(&address), sizeof(address));
        listen(server_fd, 3);

        log::info("Command server listening on port {}", ntohs(address.sin_port));

        while (true)
        {
            new_socket = accept(server_fd, reinterpret_cast<struct sockaddr *>(&address), &addrlen);
            log::info("Command client connected.");

            char buffer[1024];

            while (true)
            {
                memset(buffer, 0, sizeof(buffer));
                int bytesRead = SOCKET_READ(new_socket, buffer, sizeof(buffer), 0);

                if (bytesRead <= 0)
                {
                    log::info("Client disconnected or error.");
                    CLOSE_SOCKET(new_socket);
                    break;
                }

                std::string command(buffer);
                log::info("Received command: {}", command);

                PlayLayer *pl = PlayLayer::get();
                if (!pl || !pl->m_player1)
                {
                    std::string errorResponse = R"({"error": "Game not running or not initialized"})";
                    send(new_socket, errorResponse.c_str(), static_cast<int>(errorResponse.size()), 0);
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
                        }
                    }

                    geode::queueInMainThread([percent]
                                             { controls::loadFromPercent(percent); });
                }

                if (command.find("step") != std::string::npos)
                {
                    bool press = command.find("jump") != std::string::npos || command.find("hold") != std::string::npos;
                    controls::step(4, press);
                }

                bool died = pl->m_player1->m_isDead;
                float percent = (pl->m_player1->getPositionX() / pl->m_levelLength) * 100.0f;
                std::string response = fmt::format(R"({{"dead": {}, "percent": {}}})", died ? "true" : "false", percent);

                send(new_socket, response.c_str(), static_cast<int>(response.size()), 0);
            }
        }
    }

    void frameReceiverThread()
    {
        int server_fd;
        struct sockaddr_in address;
        int opt = 1;
        socklen_t addrlen = sizeof(address);

        server_fd = socket(AF_INET, SOCK_STREAM, 0);
#if defined(_WIN32)
        setsockopt(server_fd, SOL_SOCKET, SO_REUSEADDR, reinterpret_cast<const char *>(&opt), sizeof(opt));
#else
        setsockopt(server_fd, SOL_SOCKET, SO_REUSEADDR | SO_REUSEPORT, &opt, sizeof(opt));
#endif

        address.sin_family = AF_INET;
        address.sin_addr.s_addr = INADDR_ANY;
        address.sin_port = htons(22223);

        bind(server_fd, reinterpret_cast<struct sockaddr *>(&address), sizeof(address));
        listen(server_fd, 1);

        log::info("Frame socket listening on port 22223...");

        frameSocket = accept(server_fd, reinterpret_cast<struct sockaddr *>(&address), &addrlen);
        log::info("Frame client connected.");
    }

    void start()
    {
#if defined(_WIN32)
        WSADATA wsaData;
        WSAStartup(MAKEWORD(2, 2), &wsaData);
#endif
        std::thread(serverThreadFunction).detach();
        std::thread(frameReceiverThread).detach();
    }
} // namespace tcpserver
