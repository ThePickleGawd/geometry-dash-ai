#include <Geode/Geode.hpp>
#include <Geode/modify/PlayLayer.hpp>
#include <Geode/modify/PlayerObject.hpp>
#include <Geode/modify/CCKeyboardDispatcher.hpp>
#include <Geode/modify/AppDelegate.hpp>
#include "utils/controls.hpp"
#include <OpenGL/gl.h>
#include "utils/server.hpp"

using namespace geode::prelude;

// ============== Entry Point ==============
$on_mod(Loaded)
{
	log::info("Mod loaded, let's setup tcp server");
	tcpserver::start();
}

// ============== Stop Game from Pausing ==============
class $modify(AppDelegate)
{
	void applicationWillResignActive()
	{
		// Do nothing to avoid pausing
	}

	void applicationDidEnterBackground()
	{
		// Also override this to do nothing
	}
};

// ============== Overload jump request ==============
// class $modify(RLPlayerObject, PlayerObject)
// {
// 	bool pushButton(PlayerButton btn)
// 	{
// 		log::info("Jump button received by player!");
// 		return PlayerObject::pushButton(btn);
// 	}
// };

// ============== Main Game Loop ==============
class $modify(MyPlayLayer, PlayLayer)
{
	struct Fields
	{
		bool frameStepMode = true;
		int framesToStep = 0;
		int frame_count = 0;
		bool placedCheckpoints = false;
	};

	bool init(GJGameLevel *level, bool p1, bool p2)
	{
		log::info("Level started");

		return PlayLayer::init(level, p1, p2);
	}

	void destroyPlayer(PlayerObject *player, GameObject *p1) override
	{
		PlayLayer::destroyPlayer(player, p1);

		if (!player->m_isDead)
			return;

		log::info("player died at {} percent", (m_player1->getPositionX() / m_levelLength) * 100.0f);
	}

	void postUpdate(float p0) override
	{
		if (m_fields->frameStepMode)
		{
			if (m_fields->framesToStep > 0)
			{
				controls::unfreeze();
			}
			else
			{
				controls::freeze();
			}
			m_fields->framesToStep--;
		}
		if (m_fields->frame_count % 15 == 0)
		{
			this->sendFrameToPython();
		}
		m_fields->frame_count++;
		if (!m_fields->placedCheckpoints && m_player1 && m_checkpointArray) {
            float totalLength = m_levelLength;
            float originalX = m_player1->getPositionX();
            float originalY = m_player1->getPositionY();

            log::info("Placing premade checkpoints every 2%...");

            for (int i = 2; i < 100; i += 2) { // Skip 0% and 100%
                float targetX = (i / 100.0f) * m_levelLength;
				float offsetY = 5.0f * sin(i); // small variation

				// TODO: make sure all checkpoints are saved at safe y positions

				m_player1->setPositionX(targetX);
				m_player1->setPositionY(originalY + offsetY); // unique position

				auto checkpoint = this->createCheckpoint();
				this->storeCheckpoint(checkpoint);
            }

            // Restore player position
            m_player1->setPositionX(originalX);
            m_player1->setPositionY(originalY);

			m_fields->placedCheckpoints = true;

            log::info("Placed {} checkpoints.", m_checkpointArray->count());
        }
		PlayLayer::postUpdate(p0);
	}

	void sendFrameToPython()
	{
		// Get frame width and height
		GLint viewport[4]; // { x, y, width, height}
		glGetIntegerv(GL_VIEWPORT, viewport);
		GLint width = viewport[2];
		GLint height = viewport[3];

		// Read buffer
		unsigned char *buffer = new unsigned char[width * height * 4];
		log::info("Capturing screen of size {}x{}", width, height);
		glReadPixels(0, 0, width, height, GL_RGBA, GL_UNSIGNED_BYTE, buffer);

		// Send to AI Model via TCP
		tcpserver::sendFrame(buffer, width, height);

		delete[] buffer;
	}
};

// ============== Debugging w/ keyboard ==============
class $modify(RLCCKeyboardDispatcher, CCKeyboardDispatcher)
{
	bool dispatchKeyboardMSG(enumKeyCodes key, bool down, bool repeat)
	{
		if (down && key == cocos2d::KEY_O)
		{
			log::info("Stepping...");

			controls::step(5, false);

			return true; // Key was handled
		}
		else if (down && key == cocos2d::KEY_F)
		{
			if (auto pl = PlayLayer::get())
			{
				pl->m_player1->m_isLocked = !pl->m_player1->m_isLocked;
				if (!pl->m_player1->m_isLocked)
				{
					if (auto mypl = as<MyPlayLayer *>(pl))
					{
						mypl->m_fields->frameStepMode = false;
					}
				}
			}
		}
		else if (down && key == cocos2d::KEY_P)
		{
			if (auto pl = PlayLayer::get())
			{
				auto checkpoints = pl->m_checkpointArray;
				if (checkpoints && checkpoints->count() > 20)
				{
					int index = 1;
					auto checkpoint = static_cast<CheckpointObject*>(checkpoints->objectAtIndex(index));
					pl->m_currentCheckpoint = checkpoint;
					pl->loadFromCheckpoint(checkpoint);
					pl->destroyPlayer(pl->m_player1, nullptr); // triggers checkpoint respawn
				}
			}
		}

		// Let other keys go through
		return CCKeyboardDispatcher::dispatchKeyboardMSG(key, down, repeat);
	}
};

// ============== Step Function ==============

// I kinda hate this, but it depends on MyPlayLayer so we need to put this function here
namespace controls
{
	inline void step(int frames, bool press_jump)
	{
		if (auto pl = PlayLayer::get())
		{
			if (auto mypl = as<MyPlayLayer *>(pl))
			{
				mypl->m_fields->frameStepMode = true;
				mypl->m_fields->framesToStep = frames;

				if (mypl->m_player1)
				{
					if (press_jump)
					{
						controls::unfreeze();
						mypl->m_player1->pushButton(PlayerButton::Jump);
					}
					mypl->m_player1->lockPlayer();
				}
			}
		}
	}

}