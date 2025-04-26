#include "server.hpp"
#include <Geode/Geode.hpp>
#include <thread>
#include <string>
#include <netinet/in.h>
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
        address.sin_port = htons(12345);

        bind(server_fd, (struct sockaddr *)&address, sizeof(address));
        listen(server_fd, 3);

        log::info("TCP server started!");

        while (true)
        {
            new_socket = accept(server_fd, (struct sockaddr *)&address, (socklen_t *)&addrlen);
            char buffer[1024] = {0};
            read(new_socket, buffer, 1024);

            std::string command(buffer);
            log::info("Received command: {}", command);

            if (command.find("jump") != std::string::npos)
            {
                if (auto pl = GameManager::sharedState()->getPlayLayer())
                {
                    if (auto player = pl->m_player1)
                    {
                        player->pushButton(PlayerButton::Jump);
                    }
                }
            }

            close(new_socket);
        }
    }

    void start()
    {
        std::thread(serverThreadFunction).detach();
    }

}
