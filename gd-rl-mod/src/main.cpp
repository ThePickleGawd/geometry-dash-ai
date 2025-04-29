#include <Geode/Geode.hpp>
#include <Geode/modify/PlayLayer.hpp>
#include <Geode/modify/PlayerObject.hpp>
#include <Geode/modify/CCKeyboardDispatcher.hpp>
#include "controls.hpp"
#include <OpenGL/gl.h>
#include "tcpserver/server.hpp"

using namespace geode::prelude;

$on_mod(Loaded) {
	log::info("Mod loaded, let's setup tcp server");
	tcpserver::start();
}

class $modify(RLPlayerObject, PlayerObject) {
	bool pushButton(PlayerButton btn) {
		log::info("Jump button received by player!");
		return PlayerObject::pushButton(btn);
	}
};

class $modify(MyPlayLayer, PlayLayer)
{
	struct Fields
	{
		bool frameStepMode = false;
        int framesToStep = 0;
		int frame_count = 0;
	};

	bool init(GJGameLevel *level, bool p1, bool p2)
	{
		log::info("Level started");

		return PlayLayer::init(level, p1, p2);
	}

	void destroyPlayer(PlayerObject* player, GameObject* p1) override {
		PlayLayer::destroyPlayer(player, p1);

		if (!player->m_isDead)
			return;

		log::info("player died at {} percent", (m_player1->getPositionX() / m_levelLength) * 100.0f);
	}

	void postUpdate(float p0) override {
		if (m_fields->frameStepMode) {
			if (m_fields->framesToStep > 0) {
				controls::unfreeze();
			} else {
				controls::freeze();
			}
			m_fields->framesToStep--;
        }
		if (m_fields->frame_count % 15 == 0)
		{
			this->captureScreen();
		}
		m_fields->frame_count++;
		PlayLayer::postUpdate(p0);
	}

	void captureScreen()
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
		tcpserver::sendScreen(buffer, width, height);

		delete[] buffer;
	}
};

class $modify(RLCCKeyboardDispatcher, CCKeyboardDispatcher) {
	bool dispatchKeyboardMSG(enumKeyCodes key, bool down, bool repeat) {
		if (down && key == cocos2d::KEY_O) {
			log::info("Stepping...");

			controls::step(5, false);

			return true; // Key was handled
		} else if (down && key == cocos2d::KEY_F) {
			if (auto pl = PlayLayer::get()) {
				pl->m_player1->m_isLocked = !pl->m_player1->m_isLocked;
				if (!pl->m_player1->m_isLocked) {
					if (auto mypl = as<MyPlayLayer*>(pl)) {
						mypl->m_fields->frameStepMode = false;
					}
				}
			}
		}

		// Let other keys go through
		return CCKeyboardDispatcher::dispatchKeyboardMSG(key, down, repeat);
	}

};

namespace controls {

inline void step(int frames, bool press_jump) {
    if (auto pl = PlayLayer::get()) {
        if (auto mypl = as<MyPlayLayer*>(pl)) {
            mypl->m_fields->frameStepMode = true;
            mypl->m_fields->framesToStep = frames;

            if (mypl->m_player1) {
                if (press_jump) {
					controls::unfreeze();
                    mypl->m_player1->pushButton(PlayerButton::Jump);
                }
                mypl->m_player1->lockPlayer();
            }
        }
    }
}

}