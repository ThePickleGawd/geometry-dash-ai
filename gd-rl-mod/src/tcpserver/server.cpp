#include "server.hpp"
#include <Geode/Geode.hpp>
#include <thread>
#include <string>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <unistd.h>

using namespace geode::prelude;

namespace tcpserver
{
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

        log::info("Server listening on port {}", ntohs(address.sin_port));

        while (true)
        {
            new_socket = accept(server_fd, (struct sockaddr *)&address, (socklen_t *)&addrlen);
            char buffer[1024] = {0};
            read(new_socket, buffer, 1024);

            std::string command(buffer);
            log::info("Received command: {}", command);

            // if (command.find("jump") != std::string::npos)
            // {
            //     if (auto pl = GameManager::sharedState()->getPlayLayer())
            //     {
            //         if (auto player = pl->m_player1)
            //         {
            //             player->pushButton(PlayerButton::Jump);
            //         }
            //     }
            // }

            // TODO: Ishan
            // We have two commands: "hold" and "release"
            // They are only called when the state changes,
            // Just update the internal state, and use when we call .step

            // TODO: Ishan
            // Also we have a command "reset"
            //     TODO: Reset at a certain time position (randomly chosen by gym env)

            // TODO: step
            // Step the game based on our internal state

            const char *response = "ok";
            send(new_socket, response, strlen(response), 0);

            // close(new_socket);
        }
    }

    void start()
    {
        std::thread(serverThreadFunction).detach();
    }

    void sendScreen(unsigned char *buffer, int width, int height)
    {
        int sock = 0;
        struct sockaddr_in serv_addr;

        sock = socket(AF_INET, SOCK_STREAM, 0);
        if (sock < 0)
        {
            log::error("Socket creation error");
            return;
        }

        serv_addr.sin_family = AF_INET;
        serv_addr.sin_port = htons(22223); // Use different port for sending screen data

        if (inet_pton(AF_INET, "127.0.0.1", &serv_addr.sin_addr) <= 0)
        {
            log::error("Invalid address / Address not supported");
            return;
        }

        if (connect(sock, (struct sockaddr *)&serv_addr, sizeof(serv_addr)) < 0)
        {
            log::error("Connection Failed");
            return;
        }

        // First, send width and height (so Python knows the size)
        int meta[2] = {width, height};
        send(sock, meta, sizeof(meta), 0);

        // Then send pixel buffer
        int imageSize = width * height * 4; // 4 bytes per pixel (RGBA)
        send(sock, buffer, imageSize, 0);

        close(sock);
    }

}
