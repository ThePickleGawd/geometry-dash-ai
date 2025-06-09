#include "controls.hpp"
#include <Geode/Geode.hpp>
#include <Geode/modify/PlayLayer.hpp>
#include <Geode/modify/PlayerObject.hpp>
#include <Geode/utils/casts.hpp>
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

    void loadFromPercent(int percent)
    {
        if (auto pl = GameManager::sharedState()->getPlayLayer())
        {
            auto checkpoints = pl->m_checkpointArray;
            if (!checkpoints)
                return;
            float targetX = (percent / 100.0f) * pl->m_levelLength;

            SafeState state = {105.0f, 0, 0.0f, 0.0f}; // fallback
            if (g_safeStateMap.contains(percent))
            {
                log::info("YEAH");
                state = g_safeStateMap[percent];
            }
            else
            {
                log::info("OH NO {}", percent);
            }
            log::info("Safe state map size: {}", g_safeStateMap.size());

            auto player = pl->m_player1;
            player->setPosition({targetX, state.y});
            player->setRotation(state.rotation);
            player->m_yVelocity = state.yVelocity;

            // if ((percent >= 29.79 && percent <= 46.92) || percent >= 85.7) { // hard-coded ship intervals for stereo madness
            if (state.gamemode == 1)
            {
                player->m_isShip = true;
                player->resetPlayerIcon();
                log::info("SHIPPPPPPPPPPPPP at percent {}", percent);
            }
            else
            {
                player->m_isShip = false;
                player->resetPlayerIcon();
            }

            auto checkpoint = pl->createCheckpoint();
            pl->storeCheckpoint(checkpoint);
            pl->m_currentCheckpoint = checkpoint;

            pl->loadFromCheckpoint(checkpoint);
            player->loadFromCheckpoint(checkpoint->m_player1Checkpoint);
            resetLevel();
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

    bool isDead()
    {
        if (auto pl = GameManager::sharedState()->getPlayLayer())
        {
            return pl->m_player1 && pl->m_player1->m_isDead;
        }
        return false;
    }

} // namespace controls
