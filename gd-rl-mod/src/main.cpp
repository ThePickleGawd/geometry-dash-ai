#include <Geode/Geode.hpp>
#include <Geode/modify/PlayLayer.hpp>
#include <Geode/modify/PlayerObject.hpp>
#include <OpenGL/gl.h>
#include "tcpserver/server.hpp"

using namespace geode::prelude;

$on_mod(Loaded)
{
	log::info("Mod loaded, let's setup tcp server");
	tcpserver::start();
}

class $modify(MyPlayerObject, PlayerObject)
{
	bool pushButton(PlayerButton btn)
	{
		log::info("Jump button received by player!");
		// PlayerObject::lockPlayer();
		this->m_isLocked = !this->m_isLocked;
		return PlayerObject::pushButton(btn);
	}
};

class $modify(MyPlayLayer, PlayLayer)
{
	struct Fields
	{
		int frame_count = 0;
	};

	bool init(GJGameLevel *level, bool p1, bool p2)
	{
		log::info("Level started");

		return PlayLayer::init(level, p1, p2);
	}

	void destroyPlayer(PlayerObject *player, GameObject *p1)
	{
		PlayLayer::destroyPlayer(player, p1);

		if (!player->m_isDead)
			return;

		log::info("player died at {} percent", (m_player1->getPositionX() / m_levelLength) * 100.0f); // there's probably a field for this but idk what it is
	}

	void postUpdate(float p0)
	{
		PlayLayer::postUpdate(p0);

		if (m_fields->frame_count % 15 == 0)
		{
			this->captureScreen();
		}
		m_fields->frame_count++;
	}

	void captureScreen()
	{
		auto frameSize = CCDirector::sharedDirector()->getOpenGLView()->getFrameSize();
		int width = frameSize.width;
		int height = frameSize.height;

		unsigned char *buffer = new unsigned char[width * height * 4];
		log::info("Capturing screen of size {}x{}", width, height);

		glReadPixels(0, 0, width, height, GL_RGBA, GL_UNSIGNED_BYTE, buffer);

		tcpserver::sendScreen(buffer, width, height);

		delete[] buffer;
	}
};
