#include "controls.hpp"
#include <Geode/Geode.hpp>
#include <Geode/modify/PlayLayer.hpp>
#include <Geode/modify/PlayerObject.hpp>
#include "safe_states.hpp"

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

    void loadFromPercent(int percent) {

        if (auto pl = GameManager::sharedState()->getPlayLayer()) {
            auto checkpoints = pl->m_checkpointArray;
            if (!checkpoints) return;
            float targetX = (percent / 100.0f) * pl->m_levelLength;

            SafeState state = {105.0f, 0, 0.0f, 0.0f}; // fallback
            if (g_safeStateMap.contains(percent)) {
                state = g_safeStateMap[percent];
            }

            auto player = pl->m_player1;
            player->setPosition({targetX, state.y});
            player->setRotation(state.rotation);
            player->m_savedObjectType = static_cast<GameObjectType>(state.gamemode);
            player->m_yVelocity = state.yVelocity;

            auto checkpoint = pl->createCheckpoint();
            pl->storeCheckpoint(checkpoint);
            pl->m_currentCheckpoint = checkpoint;

            pl->loadFromCheckpoint(checkpoint);
            player->loadFromCheckpoint(checkpoint->m_player1Checkpoint);

            pl->destroyPlayer(player, nullptr);
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

    bool isDead() {
        if (auto pl = GameManager::sharedState()->getPlayLayer()) {
            return pl->m_player1 && pl->m_player1->m_isDead;
        }
        return false;
    }

} // namespace controls
