#include "controls.hpp"
#include <Geode/Geode.hpp>
#include <Geode/modify/PlayLayer.hpp>
#include <Geode/modify/PlayerObject.hpp>

using namespace geode::prelude;

namespace controls
{
    void pressJump()
    {
        if (auto pl = GameManager::sharedState()->getPlayLayer())
        {
            if (auto player = pl->m_player1)
            {
                player->pushButton(PlayerButton::Jump);
            }
        }
    }

    void releaseJump()
    {
        if (auto pl = GameManager::sharedState()->getPlayLayer())
        {
            if (auto player = pl->m_player1)
            {
                player->releaseButton(PlayerButton::Jump);
            }
        }
    }

    void resetLevel()
    {
        if (auto pl = GameManager::sharedState()->getPlayLayer())
        {
            pl->resetLevel();
        }
    }

    void freeze()
    {
        if (auto pl = GameManager::sharedState()->getPlayLayer())
        {
            if (auto player = pl->m_player1)
            {
                player->lockPlayer();
            }
        }
    }

    void unfreeze()
    {
        if (auto pl = GameManager::sharedState()->getPlayLayer())
        {
            if (auto player = pl->m_player1)
            {
                player->m_isLocked = false;
            }
        }
    }
} // namespace controls
