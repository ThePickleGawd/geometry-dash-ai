#include "utils/platform.hpp"

#include <Geode/Geode.hpp>
#include <Geode/modify/PlayLayer.hpp>
#include <Geode/modify/PlayerObject.hpp>
#include <Geode/modify/CCKeyboardDispatcher.hpp>
#include <Geode/modify/AppDelegate.hpp>
#include "utils/controls.hpp"
#include <cstdlib>
#include "utils/server.hpp"
#include "utils/safe_states.hpp"
#include <filesystem>
#include <string_view>

using namespace geode::prelude;

// ============== Entry Point ==============
std::string getSourceDir()
{
	// __FILE__ gives full path to *this* source file at compile time
	constexpr std::string_view filePath = __FILE__;
	return std::filesystem::path(filePath).parent_path().string();
}

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
		int jump_frames = 0;
		bool saveStates = false; // determines whether to save game states for checkpoint loading
		bool loadStates = true;	 // determines whether to load game states into map
	};

	bool init(GJGameLevel *level, bool p1, bool p2)
	{
		log::info("Level started");

		if (m_fields->loadStates)
		{
			std::string path = getSourceDir() + "/safe_states/stereo_madness_states.txt";
			loadSafeStatesFromFile(path);
			log::info("Safe state map size: {}", g_safeStateMap.size());
		}

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
			if (m_fields->jump_frames > 0)
			{
				controls::pressJump();
			}
			else
			{
				controls::releaseJump();
			}
			m_fields->framesToStep--;
			m_fields->jump_frames--;
		}
		if (m_fields->frame_count % 5 == 0) // You can send frames less often if there's issues
		{
			this->sendFrameToPython();
		}
		m_fields->frame_count++;

		// If we want to record our checkpoints (manually I think)
		if (m_isPracticeMode && m_player1 && m_fields->saveStates)
		{
			float percent = (m_player1->getPositionX() / m_levelLength) * 100.0f;
			int int_percent = static_cast<int>(percent);

			float currentY = m_player1->getPositionY();
			int gamemode = static_cast<int>(m_player1->m_savedObjectType);
			float rotation = m_player1->getRotation();
			float yVel = m_player1->m_yVelocity;

			auto it = g_safeStateMap.find(int_percent);
			if ((it == g_safeStateMap.end() || currentY < it->second.y) && std::abs(int_percent - percent) < 0.05)
			{
				g_safeStateMap[int_percent] = {currentY, gamemode, rotation, yVel};
			}

			// Save periodically or at 100%
			if (int_percent == 100 || m_fields->frame_count % 300 == 0)
			{
				saveSafeStatesToFile(g_safeStateMap);
			}
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
		// log::info("Capturing screen of size {}x{}", width, height);
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
			controls::loadFromPercent(92);
		}

		// Let other keys go through
		return CCKeyboardDispatcher::dispatchKeyboardMSG(key, down, repeat);
	}
};

// ============== Step Function ==============

// I kinda hate this, but it depends on MyPlayLayer so we need to put this function here
namespace controls
{
	void step(int frames, bool press_jump)
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
						// controls::unfreeze();
						// mypl->m_player1->pushButton(PlayerButton::Jump);
						mypl->m_fields->jump_frames = std::min(frames, 5);
					}
					// mypl->m_player1->lockPlayer();
				}
			}
		}
	}

}